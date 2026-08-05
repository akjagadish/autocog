# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Heuristic Decision Making domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 50). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` integer expert ratings in [0, rating_max]. The per-feature validities, n_features, and rating_max are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. All three — `n_features`, `validities`, and `rating_max` — are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=4):** [1.0, 0.9, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[5, 6, 2, 2]  B=[5, 4, 4, 4]
  trial 2: A=[5, 10, 0, 0]  B=[5, 0, 1, 1]
  trial 3: A=[8, 0, 0, 0]  B=[7, 10, 10, 10]
  trial 4: A=[8, 0, 0, 0]  B=[7, 1, 1, 1]
  trial 5: A=[0, 10, 0, 0]  B=[1, 0, 10, 10]
  trial 6: A=[0, 2, 0, 0]  B=[1, 0, 1, 1]

**Rationale:** To quantitatively dissociate the advocated theory (Mixture of TTB and WADD) from the competing theory (Mixture of TTB and Weighted Tallying), this design exploits the fact that the competing theory is entirely sign-based. Both TTB and Weighted Tallying binarize feature differences into strict wins and losses, completely ignoring the cardinal magnitude of those differences. In contrast, the advocated theory includes WADD, which integrates the exact cardinal magnitudes. We construct pairs of trials where the direction of the win on every single feature remains identical, but the magnitude of the difference changes drastically. For example, in one trial Option A might win a feature by 1 point, and in the paired trial it wins by 10 points. The competing model MUST predict the exact same choice probability across these paired trials because the feature-wise signs are unchanged. The advocated model, however, will predict large shifts in choice probabilities due to the WADD component tracking the changing magnitudes.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Human decision-making in multi-attribute choice relies on a mixture of two boundedly rational heuristics: Take-The-Best (TTB) and Weighted Additive (WADD). While TTB captures the strong reliance on the highest-validity cues by making choices based solely on the best discriminating feature, WADD integrates both cue validities and cardinal feature magnitudes across all cues. Rather than stochastically switching between these strategies, decision-makers evaluate options by integrating the normalized evidence (scores) from both heuristics into a single combined evaluation before making a choice. A parameter 'alpha' dictates the relative weight of TTB versus WADD evidence, and response noise enters through a single softmax over the mixed scores.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- validities: [(0.0, 1.0)] * n_features
- rating_max: rating_max

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    w = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Heuristic
    # Sort features by validity in descending order
    order = np.argsort(w)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
    # If no feature discriminates, they tie
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        ttb_scores = np.array([0.5, 0.5])
        
    # Weighted Additive (WADD) Heuristic
    # Integrates magnitudes and validities across all cues
    wadd_scores = stim @ w
    
    # Normalize WADD scores so they occupy a similar [0, 1] scale as TTB
    w_sum = np.sum(w)
    rmax = float(parameters["rating_max"])
    if w_sum > 0 and rmax > 0:
        wadd_scores = wadd_scores / (w_sum * rmax)
        
    # Mix the scores (evidence) rather than mixing probabilities
    alpha = float(parameters["alpha"])
    mixed_scores = alpha * ttb_scores + (1.0 - alpha) * wadd_scores
    
    # Apply a single softmax to the mixed scores
    beta = float(parameters["beta"])
    z = beta * (mixed_scores - np.max(mixed_scores))
    e = np.exp(z)
    p_mixed = e / e.sum()
    
    return p_mixed
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** Human decision-making in multi-attribute choice relies on a mixture of two boundedly rational heuristics that both utilize cue validities but process feature magnitudes differently. Rather than using full compensatory integration (WADD) or unweighted counting (Tallying), individuals draw from a mixture of 'Take-The-Best' (TTB) and 'Weighted Tallying'. TTB is a lexicographic strategy that bases the choice solely on the highest-validity cue that discriminates between the options. Weighted Tallying integrates information across all cues by binarizing feature differences into strict wins/losses and weighting these binary outcomes by their respective validities. Instead of probabilistically switching between these heuristics, decision-makers integrate the evidence (scores) from both strategies into a single combined evaluation before making a choice. A parameter 'alpha' dictates the relative weight of TTB versus Weighted Tallying evidence, and response noise enters through a single softmax over the mixed scores.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- validities: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    w = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Heuristic
    # Sort features by validity in descending order
    order = np.argsort(w)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
    # If no feature discriminates, they tie
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        ttb_scores = np.array([0.5, 0.5])
        
    # Weighted Tallying Heuristic
    # Binarize feature differences into wins/losses, then weight by validity
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    wt_scores = np.array([np.sum(a_wins * w), np.sum(b_wins * w)])
    
    # Normalize weighted tally scores so they occupy a similar [0, 1] scale as TTB
    w_sum = np.sum(w)
    if w_sum > 0:
        wt_scores = wt_scores / w_sum
        
    # Mix the scores (evidence) rather than mixing probabilities
    alpha = float(parameters["alpha"])
    mixed_scores = alpha * ttb_scores + (1.0 - alpha) * wt_scores
    
    # Apply a single softmax to the mixed scores
    beta = float(parameters["beta"])
    z = beta * (mixed_scores - np.max(mixed_scores))
    e = np.exp(z)
    p_mixed = e / e.sum()
    
    return p_mixed
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features integer expert ratings (each in [0, rating_max]) for option A on this trial.
- option_b_ratings: List of n_features integer expert ratings (each in [0, rating_max]) for option B on this trial.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Trial 1: A[1] == 6\n    is_t1 = data['option_a_ratings'].apply(lambda x: x[1] == 6)\n    # Trial 2: A[1] == 10 and A[0] == 5\n    is_t2 = data['option_a_ratings'].apply(lambda x: x[1] == 10 and x[0] == 5)\n    \n    # Trial 3: A[0] == 8 and B[1] == 10\n    is_t3 = data['option_a_ratings'].apply(lambda x: x[0] == 8) & data['option_b_ratings'].apply(lambda x: x[1] == 10)\n    # Trial 4: A[0] == 8 and B[1] == 1\n    is_t4 = data['option_a_ratings'].apply(lambda x: x[0] == 8) & data['option_b_ratings'].apply(lambda x: x[1] == 1)\n    \n    # Trial 5: A[0] == 0 and A[1] == 10\n    is_t5 = data['option_a_ratings'].apply(lambda x: x[0] == 0 and x[1] == 10)\n    # Trial 6: A[0] == 0 and A[1] == 2\n    is_t6 = data['option_a_ratings'].apply(lambda x: x[0] == 0 and x[1] == 2)\n    \n    # Compute difference in probability of choosing Option A for each pair\n    diff1 = (data[is_t2]['response'] == 0).mean() - (data[is_t1]['response'] == 0).mean()\n    diff2 = (data[is_t4]['response'] == 0).mean() - (data[is_t3]['response'] == 0).mean()\n    diff3 = (data[is_t6]['response'] == 0).mean() - (data[is_t5]['response'] == 0).mean()\n    \n    # Handle potential NaNs if a subject dataframe lacks some trials (should not happen in complete data)\n    diff1 = 0.0 if pd.isna(diff1) else diff1\n    diff2 = 0.0 if pd.isna(diff2) else diff2\n    diff3 = 0.0 if pd.isna(diff3) else diff3\n    \n    return float(diff1 + diff2 + diff3)",
  "rationale": "The experimental design consists of three pairs of trials (T1 vs T2, T3 vs T4, T5 vs T6). Within each pair, the sign of the difference between Option A and Option B on every single feature is identical, but the cardinal magnitude of the difference changes. The competing theory (Mixture of TTB and Weighted Tallying) relies entirely on the signs of feature differences, completely ignoring magnitudes. Therefore, it predicts the exact same choice probabilities for both trials in each pair, yielding an expected difference of 0. The advocated theory (Mixture of TTB and WADD) integrates cardinal magnitudes via the WADD component, predicting a substantial shift in choice probabilities toward Option A in the second trial of each pair. By summing the difference in the proportion of 'A' choices across these three pairs, we obtain a metric that is near 0 for the competing theory and highly positive for the advocated theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3785,
  "candidates_token_count": 833,
  "total_token_count": 7360
}
```
