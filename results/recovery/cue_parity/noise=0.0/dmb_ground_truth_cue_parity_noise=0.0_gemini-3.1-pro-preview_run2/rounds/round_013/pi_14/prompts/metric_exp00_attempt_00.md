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
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** This design quantitatively and qualitatively dissociates the advocated 'Sequential Evidence Accumulation' (SCA) theory from the competing 'Adaptive Strategy Selection' (TTB + Tallying mixture) theory by creating scenarios where both components of the competing theory agree, but the advocated theory predicts the exact opposite choice. The competing theory is a mixture of Take-The-Best (which relies solely on the first discriminating cue) and Tallying (which relies solely on the unweighted sum of cues). By setting validities such that the top cue favors Option A, but the second and third cues favor Option B, and the fourth and fifth cues favor Option A, we create a situation where Option A wins on TTB (Cue 1) AND wins on Tallying (3 cues to 2). Thus, the competing theory strictly predicts a preference for Option A regardless of its mixture parameter. However, SCA uses a chance-corrected non-linear weighting (e.g., max(val - 0.5)^gamma). Because the second and third cues are very close in validity to the first, their combined compensatory weight can easily exceed the combined weight of the first, fourth, and fifth cues (which include near-chance cues that are heavily discounted). Therefore, if the threshold is not immediately crossed, SCA uniquely predicts a preference for Option B, a mathematical impossibility for the competing theory.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Evidence Accumulation: Decision-makers evaluate cues sequentially in descending order of validity. Each cue provides evidence proportional to a non-linear transformation of its validity above chance. Evidence is accumulated as a running difference between the two options. If the absolute accumulated evidence exceeds a threshold, search is terminated and a choice is made based on the current evidence. If all cues are evaluated without crossing the threshold, a decision is made based on the final accumulated evidence. This allows for fast, non-compensatory decisions when top cues are highly valid, while gracefully falling back to compensatory integration when early cues are less decisive.

**Parameters:**
- theta: [0.0, 10.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale weights by transforming validity above chance, allowing better separation
    weights = np.maximum(val - 0.5, 0.001) ** gamma
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if absolute accumulated evidence reaches the threshold
            if abs(E) >= theta:
                break
            
    scores = np.array([E, -E])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Adaptive Environment-Structured Strategy Selection: Decision-makers adapt their choice strategy based on the global statistical structure of the environment. Specifically, they evaluate the dispersion (e.g., standard deviation) of cue validities. In environments with high dispersion, they probabilistically favor a non-compensatory heuristic (Take-The-Best). In environments with low dispersion, they fall back to a simpler compensatory heuristic (Equal-weights Tallying). The mixture between these strategies is dynamically governed by a logistic function of the environment's validity dispersion, with independent confidence (temperature) parameters for each strategy.

**Parameters:**
- theta: [0.0, 0.3]
- k: [1.0, 100.0]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    k = float(parameters["k"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate dispersion of validities in the environment
    dispersion = float(np.std(val))
    
    # Logistic function for strategy weight
    # Higher dispersion -> higher probability of using Take-The-Best
    w_ttb = 1.0 / (1.0 + np.exp(-k * (dispersion - theta)))
    
    # Take-The-Best (TTB) component
    cue_order = np.argsort(-val, kind="stable").tolist()
    ttb_diff = 0.0
    for j in cue_order:
        if a[j] > b[j]:
            ttb_diff = 1.0
            break
        elif b[j] > a[j]:
            ttb_diff = -1.0
            break
            
    scores_ttb = np.array([ttb_diff, -ttb_diff])
    
    # Tallying component
    # Normalized by the number of cues to ensure it's on a comparable scale to TTB (-1 to 1)
    tally_diff = (np.sum(a) - np.sum(b)) / len(val)
    scores_tally = np.array([tally_diff, -tally_diff])
    
    # Convert scores to probabilities
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    p_ttb = np.exp(z_ttb) / np.sum(np.exp(z_ttb))
    
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    p_tally = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    # Mixture prediction
    p_mix = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_tuples = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x))\n    \n    t1_mask = a_tuples == (1, 0, 0, 1, 1)\n    t2_mask = a_tuples == (0, 1, 1, 0, 0)\n    \n    sca_choices = (data.loc[t1_mask, 'response'] == 1).sum() + (data.loc[t2_mask, 'response'] == 0).sum()\n    total = t1_mask.sum() + t2_mask.sum()\n    \n    if total == 0:\n        return 0.5\n    return float(sca_choices / total)",
  "rationale": "Focuses on the critical trials (Trial 1 and its mirror, Trial 2) where both components of the competing theory (Take-The-Best and Tallying) agree on the same option, but the advocated Sequential Evidence Accumulation theory predicts the opposite option due to its non-linear weighting of validities. The metric calculates the proportion of choices that align with the unique SCA prediction on these trials."
}
```

## Usage

```json
{
  "prompt_token_count": 3656,
  "candidates_token_count": 293,
  "total_token_count": 7207
}
```
