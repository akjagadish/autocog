# metric_exp00_attempt_03

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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates Sequential Evidence Accumulation (SEA) from Parallel Cue Integration with Rank Discounting (PCIRD) by exploiting their core mechanistic difference: SEA implements a strict stopping rule (ignoring all subsequent cues once a threshold is reached), whereas PCIRD evaluates all cues in parallel but applies a rank-based exponential discount. We use 6 features with a linear validity spread to create distinct evidence trajectories. In trials where early cues oscillate in favor of A and B, SEA's accumulated evidence stays near zero, preventing early stopping and forcing integration of later cues. In contrast, PCIRD's exponential rank discounting heavily penalizes the later cues regardless of early oscillation, creating diverging predictions across the parameter space of both models.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Evidence Accumulation: Decision-makers inspect cues sequentially in order of validity, accumulating evidence for the favored option. The evidence contributed by each cue is its validity scaled by a non-linear parameter gamma. Search terminates when the absolute evidence difference reaches a threshold theta, or when all cues are exhausted. A choice is then made based on the accumulated evidence with softmax noise. This unified mechanism smoothly interpolates between Take-The-Best (low threshold), Tallying (high threshold, gamma=0), and Weighted Additive (high threshold, gamma>0).

**Parameters:**
- theta: [0.0, 3.0]
- gamma: [0.0, 1.0]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities
    v = np.power(val, gamma)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += v[j] * diff
            if abs(E) >= theta:
                break
                
    # E > 0 means A is favored; E < 0 means B is favored
    scores = np.array([E, 0.0])
    
    # Softmax choice
    z = beta * (scores - np.max(scores))
    e_vals = np.exp(z)
    p = e_vals / np.sum(e_vals)
    
    # Add lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** Parallel Cue Integration with Rank Discounting: Decision-makers process all available cues in parallel rather than strictly sequentially, but they discount the evidence provided by each cue based on its validity rank. The weight of a cue is a function of its validity (scaled non-linearly) and an exponential decay based on its rank order. This mechanism allows for a soft blending of compensatory and non-compensatory decision-making: strong rank discounting mimics Take-The-Best, while weak discounting with varying validity sensitivity smoothly interpolates between Tallying and Weighted Additive strategies, avoiding the need for a rigid probabilistic mixture.

**Parameters:**
- discount_rate: [0.0, 1.0]
- gamma: [0.0, 0.8]
- beta: [0.1, 1.5]
- epsilon: [0.0, 0.8]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    discount_rate = float(parameters["discount_rate"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for rank, j in enumerate(cue_order):
        diff = a[j] - b[j]
        weight = (val[j] ** gamma) * (discount_rate ** rank)
        E += weight * diff
        
    scores = np.array([E, 0.0])
    
    # Softmax for choice probability
    z = beta * (scores - np.max(scores))
    e_vals = np.exp(z)
    p = e_vals / np.sum(e_vals)
    
    # Apply lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
[0] rationale: Trial 4 is designed such that the most valid cue favors Option B, but the next three cues favor Option A. Under Sequential Evidence Accumulation (SEA), a low threshold will cause an immediate choice of B at cue 1, while a moderate-to-high threshold will accumulate evidence through cues 2, 3, and 4, causing the evidence to swing strongly in favor of A and trigger early stopping for A. Under Parallel Cue Integration with Rank Discounting (PCIRD), the exponential decay on cue weights severely penalizes cues 2, 3, and 4. Thus, PCIRD heavily favors B across most of its parameter space unless the discount rate is extremely close to 1. This produces a reliable divergence in the probability of choosing A on this specific trial.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 4 where early cue favors B, but a streak of subsequent cues favors A
    # Trial 4: A=[0, 1, 1, 1, 0, 0], B=[1, 0, 0, 0, 1, 1]
    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 0, 0))
    if not t4_mask.any():
        return 0.5
    
    # Return the proportion of times option A was chosen in Trial 4
    return float(np.mean(data.loc[t4_mask, 'response'] == 0))
outcome: self_sim=0.5175 (var=0.0524) adversary_sim=0.4838 (var=0.0215) welch_t=+0.621 p=0.5382 (N=25, alpha=0.01) -> reject

[1] rationale: Measures the variance of the choice probabilities across the 6 distinct trials. Sequential Evidence Accumulation (SEA) often triggers early stopping based on the first few cues, leading to more extreme, polarized choice probabilities across different trial types. In contrast, Parallel Cue Integration with Rank Discounting (PCIRD) integrates all cues (albeit with discounting), which softens the evidence accumulation and typically pulls choice probabilities closer to the center. Consequently, the cross-trial variance in choice proportions should be systematically higher for SEA than for PCIRD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['trial_id'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    trial_means = data.groupby('trial_id')['response'].mean()
    return float(trial_means.var(ddof=0))
outcome: self_sim=0.0179 (var=0.0014) adversary_sim=0.0032 (var=0.0002) welch_t=+1.883 p=0.0694 (N=25, alpha=0.01) -> reject

[2] rationale: Measures the difference in preference for Option A between Trial 2 and Trial 5. In Trial 2, Option A is supported by the two most valid cues, making it a very strong early candidate for SEA, which often stops before considering the four opposing cues. In Trial 5, A is supported by cues 1, 3, and 4. PCIRD integrates these with rank discounting, which treats the two profiles differently than SEA's threshold-based early stopping, leading to a diverging difference in choice probabilities between the two trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0, 0))
    t5_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 1, 0, 0))
    if not t2_mask.any() or not t5_mask.any():
        return 0.0
    p_a_t2 = np.mean(data.loc[t2_mask, 'response'] == 0)
    p_a_t5 = np.mean(data.loc[t5_mask, 'response'] == 0)
    return float(p_a_t2 - p_a_t5)
outcome: self_sim=-0.1162 (var=0.0559) adversary_sim=-0.0150 (var=0.0326) welch_t=-1.701 p=0.09577 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 1, 0))\n    t3_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1, 1))\n    if not t1_mask.any() or not t3_mask.any():\n        return 0.0\n    p_a_t1 = np.mean(data.loc[t1_mask, 'response'] == 0)\n    p_a_t3 = np.mean(data.loc[t3_mask, 'response'] == 0)\n    return float(p_a_t3 - p_a_t1)",
  "rationale": "Measures the difference in preference for Option A between Trial 3 and Trial 1. In Trial 1, evidence oscillates and ultimately nets to near zero, so Sequential Evidence Accumulation (SEA) with a high threshold will integrate all cues and effectively guess. In Trial 3, early evidence oscillates but later cues strongly favor A, so SEA with a high threshold will integrate to a strong preference for A. Thus, SEA predicts a significantly higher probability of choosing A in Trial 3 than Trial 1. In contrast, PCIRD heavily discounts later cues, relying primarily on the early cues which favor A in both trials, predicting little to no difference."
}
```

## Usage

```json
{
  "prompt_token_count": 4330,
  "candidates_token_count": 351,
  "total_token_count": 34106
}
```
