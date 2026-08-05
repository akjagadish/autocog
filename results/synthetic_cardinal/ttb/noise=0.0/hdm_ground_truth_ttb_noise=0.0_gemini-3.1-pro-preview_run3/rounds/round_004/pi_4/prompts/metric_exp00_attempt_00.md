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
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[6, 4, 4, 4]  B=[5, 5, 5, 5]
  trial 2: A=[10, 4, 4, 4]  B=[0, 5, 5, 5]
  trial 3: A=[5, 5, 5, 5]  B=[4, 6, 6, 6]
  trial 4: A=[5, 2, 2, 2]  B=[4, 8, 8, 8]
  trial 5: A=[5, 5, 5, 5]  B=[6, 4, 4, 4]
  trial 6: A=[0, 5, 5, 5]  B=[10, 4, 4, 4]
  trial 7: A=[8, 4, 4, 4]  B=[7, 5, 5, 5]
  trial 8: A=[2, 5, 5, 5]  B=[1, 6, 6, 6]

**Rationale:** This design quantitatively dissociates the advocated theory (Mixture of TTB and Weighted Tallying) from the competing theory (Mixture of TTB and WADD) by exploiting the fact that the advocated theory is entirely sign-based. Both TTB and Weighted Tallying binarize feature differences into strict wins and losses, completely ignoring the cardinal magnitude of those differences. In contrast, the competing theory includes WADD, which integrates the exact cardinal magnitudes. We construct pairs of trials where the direction of the win on every single feature remains identical, but the magnitude of the difference changes drastically. For example, in one trial Option A might win the highest-validity feature by 1 point and lose the others by 1 point, leading WADD to prefer B. In the paired trial, Option A wins the highest-validity feature by 10 points and loses the others by 1 point, causing WADD to strongly prefer A. The advocated model MUST predict the exact same choice probability across these paired trials because the feature-wise signs are unchanged. The competing model, however, will predict large shifts in choice probabilities due to the WADD component tracking the changing magnitudes.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\ndef metric(data: pd.DataFrame) -> float:\n    a_ratings = np.vstack(data['option_a_ratings'].values)\n    b_ratings = np.vstack(data['option_b_ratings'].values)\n    \n    choose_a = 1 - data['response'].values\n    \n    # Trial 2: Option A wins the first feature by exactly 10 points\n    mask_t2 = (a_ratings[:, 0] - b_ratings[:, 0]) == 10\n    \n    # Trial 4: Option B wins the second feature by exactly 6 points\n    mask_t4 = (b_ratings[:, 1] - a_ratings[:, 1]) == 6\n    \n    if mask_t2.sum() == 0 or mask_t4.sum() == 0:\n        return 0.0\n        \n    p_a_t2 = choose_a[mask_t2].mean()\n    p_a_t4 = choose_a[mask_t4].mean()\n    \n    return float(p_a_t2 - p_a_t4)",
  "rationale": "This metric calculates the difference in the probability of choosing Option A between Trial 2 and Trial 4. In both of these trials, the direction of the feature differences is identical: Option A wins the highest validity feature, while Option B wins the remaining three features. According to the advocated theory (TTB + Weighted Tallying), which relies exclusively on the signs of feature differences, the probability of choosing Option A should be identical across these two trials, yielding an expected metric value of 0. However, the competing theory incorporates WADD, which is sensitive to the cardinal magnitudes of these differences. In Trial 2, Option A wins the first feature by a massive 10 points, leading WADD to strongly favor A. In Trial 4, Option B wins the other features by 6 points each, leading WADD to strongly favor B. Consequently, the competing theory will predict a large positive difference for this metric."
}
```

## Usage

```json
{
  "prompt_token_count": 3889,
  "candidates_token_count": 487,
  "total_token_count": 8638
}
```
