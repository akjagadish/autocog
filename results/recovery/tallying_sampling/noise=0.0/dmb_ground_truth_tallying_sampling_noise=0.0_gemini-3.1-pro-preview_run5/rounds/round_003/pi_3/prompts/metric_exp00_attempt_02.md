# metric_exp00_attempt_02

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
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 3: A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 7: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 8: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 9: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 10: A=[1, 0, 1, 0]  B=[0, 1, 1, 0]

**Rationale:** To quantitatively dissociate pure Tallying (Equal Weight) from Tallying with Lexicographic Tie-Breaking, we focus on trials where the two options have an equal number of positive features (i.e., their tallies are tied) but differ in which specific features are positive. Pure Tallying predicts exact indifference (50/50 choice) on these tie trials, regardless of the validities of the features. In contrast, Tallying with Lexicographic Tie-Breaking resolves these ties by selecting the option that is favored by the single most valid discriminating feature. By including a mix of tie trials (where the models strongly diverge) and non-tie trials (where both models predict the same choice based on the tally), we can reliably estimate choice parameters and test for the presence of the validity-based tie-breaking mechanism.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Tallying (Equal Weight) theory posits that decision-makers simply count the number of positive features (or cues) for each option and choose the option with the higher tally, ignoring cue validities completely. This is a compensatory heuristic that treats all pieces of evidence equally.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Stimulus is the pair of option feature vectors for the current trial
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Tallying with Lexicographic Tie-Breaking posits that decision-makers primarily rely on a compensatory equal-weight tallying heuristic, choosing the option with the highest number of positive features. However, when options are tied in their feature tallies, decision-makers do not guess randomly. Instead, they break the tie by comparing the options on the single most valid feature where the options differ (a Take-The-Best mechanism). This hybrid approach perfectly mimics Tallying on unequal feature counts but provides a deterministic, validity-based resolution for ties.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- w_tie: [0.0, 0.05]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
        
    # Primary mechanism: Equal-weight tallying
    scores = np.sum(stim, axis=1)
    
    # Secondary mechanism: Lexicographic tie-breaking (Take-The-Best)
    if scores[0] == scores[1]:
        validities = np.asarray(parameters["validities"], dtype=float)
        w_tie = float(parameters["w_tie"])
        diff = stim[0] - stim[1]
        valid_diffs = np.where(diff != 0)[0]
        
        if len(valid_diffs) > 0:
            # Find the differing feature with the highest validity
            best_feature = valid_diffs[np.argmax(validities[valid_diffs])]
            if diff[best_feature] > 0:
                scores[0] += w_tie
            else:
                scores[1] += w_tie

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
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
[0] rationale: This metric isolates trials where the two options have an equal total number of positive features (i.e., a tally tie). Under pure Tallying, subjects should guess randomly on these trials (50% match rate with any deterministic tie-breaking rule). Under Tallying with Lexicographic Tie-Breaking, subjects resolve the tie by choosing the option favored by the single most valid differing feature, resulting in a match rate significantly above 50%.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    score_a = np.sum(a_ratings, axis=1)
    score_b = np.sum(b_ratings, axis=1)
    
    tie_mask = (score_a == score_b)
    if not np.any(tie_mask):
        return 0.5
        
    a_ties = a_ratings[tie_mask]
    b_ties = b_ratings[tie_mask]
    responses_ties = data['response'].values[tie_mask]
    
    diff = a_ties - b_ties
    
    matches = 0
    valid_ties = 0
    for i in range(len(diff)):
        d = diff[i]
        valid_diffs = np.where(d != 0)[0]
        if len(valid_diffs) > 0:
            # The validities are [0.9, 0.8, 0.7, 0.6], strictly decreasing.
            # Thus, the first non-zero difference corresponds to the most valid differing feature.
            best_feature = valid_diffs[0]
            lex_winner = 0 if d[best_feature] > 0 else 1
            if responses_ties[i] == lex_winner:
                matches += 1
            valid_ties += 1
            
    return matches / valid_ties if valid_ties > 0 else 0.5
outcome: self_sim=0.5073 (var=0.0028) adversary_sim=0.5362 (var=0.0044) welch_t=-1.700 p=0.0959 (N=25, alpha=0.01) -> reject

[1] rationale: To drastically reduce between-subject variance compared to a simple match rate, this metric computes the covariance between the subject's choices and the lexicographic tie-breaker predictions strictly on tie trials. By centering the choices, it automatically corrects for any individual baseline bias toward option A or B (e.g., spatial biases). Under pure Tallying, choices on tie trials are independent of the lexicographic direction, yielding an expected covariance of 0. Under Tallying with Lexicographic Tie-Breaking, choices are positively correlated with the tie-breaker, yielding a positive covariance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    score_a = np.sum(a_ratings, axis=1)
    score_b = np.sum(b_ratings, axis=1)
    
    tie_mask = (score_a == score_b)
    if not np.any(tie_mask):
        return 0.0
        
    a_ties = a_ratings[tie_mask]
    b_ties = b_ratings[tie_mask]
    
    # Map responses: 1 for choosing A, -1 for choosing B
    responses = data['response'].values[tie_mask]
    y = np.where(responses == 0, 1.0, -1.0)
    
    diff = a_ties - b_ties
    x = np.zeros(len(diff))
    
    for i in range(len(diff)):
        d = diff[i]
        valid_diffs = np.where(d != 0)[0]
        if len(valid_diffs) > 0:
            # Validities are strictly decreasing, so the first non-zero difference is the most valid
            best_feature = valid_diffs[0]
            x[i] = 1.0 if d[best_feature] > 0 else -1.0
            
    if len(x) < 2:
        return 0.0
        
    # Return the sample covariance between the lexicographic prediction and the actual choice
    return float(np.cov(x, y)[0, 1])
outcome: self_sim=-0.0031 (var=0.0061) adversary_sim=0.0397 (var=0.0072) welch_t=-1.861 p=0.06893 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_ratings = np.stack(data['option_a_ratings'].values)\n    b_ratings = np.stack(data['option_b_ratings'].values)\n    \n    score_a = np.sum(a_ratings, axis=1)\n    score_b = np.sum(b_ratings, axis=1)\n    \n    tie_mask = (score_a == score_b)\n    if not np.any(tie_mask):\n        return 0.0\n        \n    diff = a_ratings - b_ratings\n    responses = data['response'].values\n    chose_A = (responses == 0).astype(float)\n    \n    lex_favors_A = []\n    lex_favors_B = []\n    \n    for i in range(len(diff)):\n        if tie_mask[i]:\n            d = diff[i]\n            valid_diffs = np.where(d != 0)[0]\n            if len(valid_diffs) > 0:\n                # Validities are strictly decreasing, so the first non-zero diff is the most valid\n                best_feature = valid_diffs[0]\n                if d[best_feature] > 0:\n                    lex_favors_A.append(chose_A[i])\n                else:\n                    lex_favors_B.append(chose_A[i])\n                    \n    if len(lex_favors_A) == 0 or len(lex_favors_B) == 0:\n        return 0.0\n        \n    # Difference in probability of choosing Option A when Lexicographic tie-breaker favors A vs B\n    return float(np.mean(lex_favors_A) - np.mean(lex_favors_B))\n",
  "rationale": "To drastically reduce between-subject variance and amplify the effect size, this metric separates tie trials into those where the lexicographic tie-breaker favors Option A and those where it favors Option B. It then computes the difference in the subject's empirical probability of choosing Option A between these two sets. Under pure Tallying, the choice is independent of the lexicographic direction, so the expected difference is 0. Under Tallying with Lexicographic Tie-Breaking, subjects will choose A more often when the tie-breaker favors A and less often when it favors B, resulting in a positive difference. Crucially, this simple difference of means perfectly cancels out any individual spatial bias (e.g., a baseline preference for choosing the left option) without suffering from the attenuation issues of a covariance computation on an imbalanced design."
}
```

## Usage

```json
{
  "prompt_token_count": 4410,
  "candidates_token_count": 638,
  "total_token_count": 7856
}
```
