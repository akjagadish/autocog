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
**Validities (n_features=8):** [0.9, 0.9, 0.9, 0.85, 0.85, 0.85, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  trial 2: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 1]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  trial 6: A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1, 0, 1]
  trial 7: A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  trial 8: A=[1, 1, 1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]

**Rationale:** This design systematically dissociates Inverse Validity Weighting (IVW) from Anti-Expertise Tallying (AET) by exploiting their different mechanisms for discounting high-validity features. IVW retains all features but assigns them subjective weights inversely proportional to their stated validity (1 - validity). Thus, a large number of high-validity features can collectively outweigh a single low-validity feature if their summed inverse-validity weights are sufficiently high. In contrast, AET applies a hard threshold, completely filtering out features whose validity exceeds the subject's personal limit before tallying the remaining features. By pitting an option with many high-validity wins against an option with fewer (but lower-validity) wins, IVW will favor the option with numerous high-validity features due to the accumulated inverse weight. However, AET will entirely discard those high-validity features for subjects with moderate-to-low thresholds, causing it to deterministically choose the option supported by the remaining low-validity features.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Anti-Expertise Tallying: Decision-makers actively distrust or ignore high-validity 'expert' cues, perceiving them as overly salient, redundant, or manipulative. Instead of using them, subjects filter out features whose stated validity exceeds a certain personal threshold. After discarding these high-validity cues, subjects apply a standard Tallying heuristic (counting strict feature-wise wins) on the remaining lower-validity features to make their choice.

**Parameters:**
- threshold: [0.5, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Anti-Expertise Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Filter out features with validity above the threshold
    mask = val <= threshold
    
    # Fallback to all features if the threshold is so low that all features are discarded
    if not np.any(mask):
        mask = np.ones_like(val, dtype=bool)
        
    a_masked = a[mask]
    b_masked = b[mask]
    
    # Tallying on the remaining features
    a_wins = float(np.sum(a_masked > b_masked))
    b_wins = float(np.sum(b_masked > a_masked))
    
    scores = np.array([a_wins, b_wins])
    
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
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Inverse Validity Weighting: Subjects actively distrust expert ratings, treating high-validity cues as manipulative or overly salient. Instead of relying on them, they use a weighted additive model where the subjective weight of each feature is inversely proportional to its stated validity (1 - validity). This mechanism leads subjects to consistently prefer options that are supported by lower-validity features over those supported by high-validity features, explaining the strong anti-expert choices observed in the experiments.

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
        raise ValueError(f"Inverse Validity Weighting expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Primary mechanism: Weighted additive using inverse validities
    # Lower validity features receive higher subjective weights
    weights = 1.0 - val
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
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
    probabilities = probabilities / probabilities.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_sums = data['option_a_ratings'].apply(sum)\n    b_sums = data['option_b_ratings'].apply(sum)\n    \n    mask_orig = (a_sums == 6) & (b_sums == 1)\n    mask_swap = (a_sums == 1) & (b_sums == 6)\n    mask_all = mask_orig | mask_swap\n    \n    if not mask_all.any():\n        return 0.5\n        \n    chose_1_orig = mask_orig & (data['response'] == 1)\n    chose_1_swap = mask_swap & (data['response'] == 0)\n    \n    return float((chose_1_orig | chose_1_swap)[mask_all].mean())",
  "rationale": "Focuses on Trial 3, which is uniquely identifiable by the options having sum scores of 6 and 1. In this trial, Inverse Validity Weighting strongly prefers the option with a sum of 6 (accumulated inverse weights of 0.75 vs 0.5). Conversely, Anti-Expertise Tallying discards the high-validity features, leaving only the low-validity features where the option with a sum of 1 has an advantage, leading it to prefer the option with a sum of 1 for thresholds < 0.85. Tracking the proportion of choices for the sum=1 option provides a stark contrast between the two models."
}
```

## Usage

```json
{
  "prompt_token_count": 3489,
  "candidates_token_count": 363,
  "total_token_count": 8393
}
```
