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
**Validities (n_features=6):** [0.9, 0.85, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=11):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  trial 6: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 10: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Softmax Tallying from Noisy Encoding Tallying by manipulating the total number of positive features while keeping the tally difference constant. Softmax Tallying predicts that choice probability depends exclusively on the difference in tallies (e.g., a 1 vs 0 trial yields the exact same choice probability as a 6 vs 5 trial, as the difference is 1 in both). In contrast, Noisy Encoding Tallying assumes each feature has an independent probability of being misperceived. Because features are bounded between 0 and 1, the binomial noise distributions become highly skewed near the boundaries (e.g., an option with 0 positive features can only gain perceived features, while an option with 6 positive features can only lose them). This causes Noisy Encoding Tallying to predict systematically different choice probabilities for 1v0 compared to 6v5, despite the constant difference. By including multiple levels of constant differences (diff=1, diff=2, diff=3) at varying absolute feature counts, we can clearly separate the exact-difference logistic predictions of Softmax Tallying from the boundary-skewed predictions of Noisy Encoding Tallying.

**Computed schedule:** 11 unique pairs × 8 reps = 88 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by tallying the total number of positive features for each option, ignoring cue validities entirely. The option with the higher unweighted sum of positive features is chosen. This Equal Weight (or Tallying) heuristic provides a frugal but fully compensatory strategy, capturing the strong human tendency to prefer options with multiple supporting cues over those with a single high-validity cue. Response noise is modeled via a softmax over the tallied scores with inverse temperature beta, and an independent lapse rate epsilon.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features (unweighted sum) for each option.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution.
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
**Description:** Noisy Encoding Tallying: Decision-makers rely on the unweighted Tallying heuristic but suffer from noisy perception or encoding of the environment. Each binary feature has an independent probability of being misperceived (a 1 flipped to a 0, or a 0 flipped to a 1). Subjects then compute the tally of these perceived features and deterministically choose the option with the higher tally, breaking ties randomly. This naturally predicts that decision errors scale with the total number of features (capturing non-linear log-odds in certain experiments) because more features provide more opportunities for bit-flips to alter the tally difference.

**Parameters:**
- flip_prob: [0.0, 0.2]
- epsilon: [0.0, 0.2]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    import math
    
    stim = np.asarray(state, dtype=float)
    n_features = stim.shape[1]
    
    p = float(parameters["flip_prob"])
    epsilon = float(parameters["epsilon"])
    
    def binom_pmf(k, n, prob):
        if n == 0:
            return 1.0 if k == 0 else 0.0
        if prob == 0.0:
            return 1.0 if k == 0 else 0.0
        if prob == 1.0:
            return 1.0 if k == n else 0.0
        return math.comb(n, k) * (prob ** k) * ((1 - prob) ** (n - k))
        
    def get_tally_dist(N1, N0, p):
        dist = np.zeros(N1 + N0 + 1)
        for x in range(N1 + 1):
            px = binom_pmf(x, N1, 1 - p)
            if px == 0.0:
                continue
            for y in range(N0 + 1):
                py = binom_pmf(y, N0, p)
                if py > 0.0:
                    dist[x + y] += px * py
        return dist
        
    N1_A = int(np.sum(stim[0]))
    N0_A = n_features - N1_A
    dist_A = get_tally_dist(N1_A, N0_A, p)
    
    N1_B = int(np.sum(stim[1]))
    N0_B = n_features - N1_B
    dist_B = get_tally_dist(N1_B, N0_B, p)
    
    prob_A_wins = 0.0
    prob_tie = 0.0
    for vA in range(len(dist_A)):
        if dist_A[vA] == 0.0:
            continue
        for vB in range(len(dist_B)):
            if dist_B[vB] == 0.0:
                continue
            if vA > vB:
                prob_A_wins += dist_A[vA] * dist_B[vB]
            elif vA == vB:
                prob_tie += dist_A[vA] * dist_B[vB]
                
    p_A_core = prob_A_wins + 0.5 * prob_tie
    p_B_core = 1.0 - p_A_core
    
    p_core = np.array([p_A_core, p_B_core])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Calculate the unweighted sum of positive features for each option\n    sum_a = data['option_a_ratings'].apply(sum)\n    sum_b = data['option_b_ratings'].apply(sum)\n    \n    # Calculate the difference in tallies (A always has more or equal positive features in this design)\n    diff = sum_a - sum_b\n    \n    # Calculate the proportion of times Option A (the better option) was chosen\n    # response == 0 indicates Option A was chosen\n    p_a_diff_1 = 1.0 - data.loc[diff == 1, 'response'].mean()\n    p_a_diff_3 = 1.0 - data.loc[diff == 3, 'response'].mean()\n    \n    # Handle edge cases where a subject might have missing data\n    if np.isnan(p_a_diff_1) or np.isnan(p_a_diff_3):\n        return 0.0\n        \n    # Return the increase in accuracy when the tally difference grows from 1 to 3\n    return float(p_a_diff_3 - p_a_diff_1)",
  "rationale": "Softmax Tallying's parameter distribution (beta in [0.1, 20.0]) heavily favors large values where the choice probability hits the ceiling (P~1.0) even for a minimal tally difference of 1. Consequently, it predicts almost no marginal increase in accuracy when the tally difference increases from 1 to 3. In contrast, Noisy Encoding Tallying's noise parameter (flip_prob up to 0.2) keeps the accuracy for a difference of 1 well below the ceiling (around 0.78 on average), allowing for a substantial and measurable increase in accuracy when the difference is 3 (around 0.99). Therefore, the difference in choice probability between diff=3 and diff=1 trials will be near zero for Softmax Tallying but large and positive for Noisy Encoding Tallying, providing a highly discriminative contrast."
}
```

## Usage

```json
{
  "prompt_token_count": 3786,
  "candidates_token_count": 503,
  "total_token_count": 32611
}
```
