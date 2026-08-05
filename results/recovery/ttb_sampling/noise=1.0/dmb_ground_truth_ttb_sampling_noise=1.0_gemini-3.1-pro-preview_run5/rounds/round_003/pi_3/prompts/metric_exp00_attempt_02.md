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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design contrasts Probabilistic Cue Sampling (PCS) and Weighted Additive (WADD) models by exploiting the mathematical properties of their validity transformations. PCS uses a softmax function over validities, which is strictly convex (exp(gamma * v)). Consequently, on trials where the sum of validities is perfectly tied, PCS will systematically assign a higher probability to the option supported by the more extreme validities (e.g., 0.9 and 0.5 over 0.8 and 0.6). WADD, however, uses a power transformation (v^gamma), which can be convex, linear, or concave depending on its gamma parameter. If gamma < 1, WADD will actually prefer the option with intermediate validities, creating a qualitative divergence from PCS. We include several such 'tied-sum' pairs at different validity levels to test this convexity versus concavity distinction, alongside compensatory pairs to ensure broad model identifiability.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Cue Sampling with Softmax Validities: Decision-makers evaluate options by sampling a single feature on each trial. The probability of sampling a feature is determined by a softmax function over the objective validities scaled by a sensitivity parameter gamma. The option that is superior on the sampled feature is chosen; ties result in guessing. This predicts intermediate choice proportions when validities are mixed, avoiding deterministic winner-takes-all behavior.

**Parameters:**
- gamma: [0.0, 2.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])

    # Compute subjective validities using softmax for sampling probabilities
    z = gamma * validities
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    p_feat = e / np.sum(e)

    a, b = stim[0], stim[1]

    # Identify wins and ties for each feature
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    ties = (a == b).astype(float)

    # Analytically marginalize over the sampled feature:
    # P(Choose A) = sum_i P(sample i) * P(Choose A | sample i)
    # P(Choose A | sample i) = 1 if a_i > b_i, 0 if b_i > a_i, 0.5 if a_i == b_i
    p_A = np.sum(p_feat * (a_wins + 0.5 * ties))
    p_B = np.sum(p_feat * (b_wins + 0.5 * ties))

    p_core = np.array([p_A, p_B])

    # Apply uniform lapse rate
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


## COMPETING THEORY
**Description:** Noisy Weighted Additive (WADD) with Regularized Non-linear Weighting: Decision-makers compute a global utility for each option by summing the subjective weights of all cues that favor it. Subjective weights are formed by applying a non-linear transformation to the objective validities (via an exponent gamma), allowing decision-makers to either amplify or compress the relative importance of high-validity cues. To prevent extreme lexicographic behavior, the degree of non-linear amplification and the choice determinism are bounded. These utilities are then translated into choice probabilities via a softmax function parameterized by an inverse temperature and a lapse rate.

**Parameters:**
- beta: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 3.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])

    # Compute a global utility for each option by summing the non-linearly weighted validities
    weights = validities ** gamma
    utilities = stim @ weights

    # Translate utilities into choice probabilities using a softmax function
    z = beta * utilities
    z = z - np.max(z)  # For numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)

    # Apply lapse rate
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

    return p_final
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
[0] rationale: This metric calculates the proportion of times the 'extreme' option is chosen in trials where both options have exactly two positive features (tied in quantity). The 'extreme' option is defined as the one possessing the single highest-validity feature among the two options. Because Probabilistic Cue Sampling (PCS) uses a strictly convex softmax transformation over validities, it systematically favors the extreme option. In contrast, the Weighted Additive (WADD) model uses a power transformation that is often concave (when gamma < 1), which penalizes extreme validities in favor of intermediate ones. Consequently, PCS will predict a higher choice proportion for the extreme option compared to WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    mask = (sum_a == 2) & (sum_b == 2)
    
    if not np.any(mask):
        return 0.5
        
    a_masked = a_ratings[mask]
    b_masked = b_ratings[mask]
    responses = data['response'].values[mask]
    
    a_first_1 = np.argmax(a_masked, axis=1)
    b_first_1 = np.argmax(b_masked, axis=1)
    
    a_is_extreme = a_first_1 < b_first_1
    b_is_extreme = b_first_1 < a_first_1
    
    extreme_chosen = 0
    total = 0
    for i in range(len(responses)):
        if a_is_extreme[i]:
            if responses[i] == 0:
                extreme_chosen += 1
            total += 1
        elif b_is_extreme[i]:
            if responses[i] == 1:
                extreme_chosen += 1
            total += 1
            
    if total == 0:
        return 0.5
        
    return extreme_chosen / total
outcome: self_sim=0.4987 (var=0.0036) adversary_sim=0.4937 (var=0.0061) welch_t=+0.254 p=0.8003 (N=25, alpha=0.01) -> reject

[1] rationale: Focuses exclusively on Trial 6, where Option 1 (0,1,1,0,0) has a higher sum of validities (1.5) but a lower maximum validity (0.8) than Option 2 (1,0,0,0,1), which has a sum of 1.4 and a max of 0.9. Because Probabilistic Cue Sampling (PCS) uses a strictly convex softmax function over validities, it is heavily influenced by the maximum validity and will systematically prefer Option 2. In contrast, the Weighted Additive (WADD) model, which evaluates the sum of non-linearly transformed validities, generally prefers Option 1 across its parameter range (especially near gamma=1 where it acts linearly). By calculating the proportion of times Option 1 is chosen in this specific trial, we obtain a stark divergence: WADD predicts a proportion > 0.5, whereas PCS predicts a proportion < 0.5.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    t1 = "01100"
    t2 = "10001"
    
    mask1 = (a_str == t1) & (b_str == t2)
    mask2 = (a_str == t2) & (b_str == t1)
    
    chosen_t1 = 0
    total = 0
    
    if mask1.any():
        chosen_t1 += (data.loc[mask1, 'response'] == 0).sum()
        total += mask1.sum()
        
    if mask2.any():
        chosen_t1 += (data.loc[mask2, 'response'] == 1).sum()
        total += mask2.sum()
        
    if total == 0:
        return 0.5
        
    return float(chosen_t1 / total)
outcome: self_sim=0.5000 (var=0.0203) adversary_sim=0.5433 (var=0.0248) welch_t=-1.021 p=0.3126 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    a_str = data['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    b_str = data['option_b_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    \n    # Trial 7: A=[1,0,0,0,0], B=[0,1,1,1,1]\n    # Trial 8: A=[0,0,1,1,0], B=[1,0,0,0,1]\n    \n    mask_t7 = (a_str == \"10000\") & (b_str == \"01111\")\n    mask_t7_rev = (a_str == \"01111\") & (b_str == \"10000\")\n    \n    mask_t8 = (a_str == \"00110\") & (b_str == \"10001\")\n    mask_t8_rev = (a_str == \"10001\") & (b_str == \"00110\")\n    \n    b_chosen = 0\n    total = 0\n    \n    if mask_t7.any():\n        b_chosen += (data.loc[mask_t7, 'response'] == 1).sum()\n        total += mask_t7.sum()\n    if mask_t7_rev.any():\n        b_chosen += (data.loc[mask_t7_rev, 'response'] == 0).sum()\n        total += mask_t7_rev.sum()\n        \n    if mask_t8.any():\n        b_chosen += (data.loc[mask_t8, 'response'] == 1).sum()\n        total += mask_t8.sum()\n    if mask_t8_rev.any():\n        b_chosen += (data.loc[mask_t8_rev, 'response'] == 0).sum()\n        total += mask_t8_rev.sum()\n        \n    if total == 0:\n        return 0.5\n        \n    return float(b_chosen / total)",
  "rationale": "This metric evaluates the choice proportion for Option B across Trials 7 and 8. In Trial 7, Option B combines four moderate-to-low validities against Option A's single highest validity. In Trial 8, Option B possesses the highest and lowest validities, while Option A has two intermediate ones. Across its entire parameter range (gamma in [0.1, 3.0]), the Weighted Additive (WADD) model consistently assigns a higher global utility to Option B in both trials, predicting a robustly high choice proportion. In contrast, Probabilistic Cue Sampling (PCS) is mathematically constrained by its strictly convex softmax function and feature-sampling nature; it predicts a substantially lower choice probability for Option B in Trial 7 (since the single highest cue heavily dominates the sampling space) and is nearly indifferent in Trial 8. By averaging these two specific trials, we isolate a consistent, low-variance preference in WADD that diverges strongly from the mathematically constrained upper bound of PCS."
}
```

## Usage

```json
{
  "prompt_token_count": 4475,
  "candidates_token_count": 777,
  "total_token_count": 36705
}
```
