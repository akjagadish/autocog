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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** This design quantitatively dissociates the Strategy Mixture model (Take-The-Best + Tallying) from the Sequential Evidence Accumulation (SEA) model. We use 5 features with linearly spaced validities (0.9, 0.8, 0.7, 0.6, 0.5). The key dissociation occurs in trials where Tallying predicts a perfect tie (e.g., Option A wins on cues 1 and 4; Option B wins on cues 2 and 3). In the Strategy Mixture model, the tie in Tallying means the decision defaults entirely to TTB, which strictly selects Option A based on the highest validity cue. However, for the SEA model, the evidence sum for Option A (0.9 + 0.6) exactly equals Option B (0.8 + 0.7). Thus, SEA predicts a perfect tie (if the threshold is high enough to integrate all cues), regardless of its non-linearity parameter gamma. Additional trials pit a Tallying advantage (e.g., 4 wins to 1) against a TTB advantage to test SEA's early stopping rule (low threshold) against the Mixture's probabilistic weighting.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Mixture (Take-The-Best and Tallying with Tallying Bias): Decision-makers are heterogeneous in their strategy use, probabilistically switching between strategies. On any given choice, a subject has a probability 'p_ttb' of applying a one-reason lexicographic heuristic (Take-The-Best) and a probability '1 - p_ttb' of applying an equal-weight compensatory heuristic (Tallying). Empirical data suggests that people generally favor Tallying over Take-The-Best in these environments, so the probability of using Take-The-Best is bounded between 10% and 50%, ensuring the mixture slightly favors Tallying to better match observed aggregate choice probabilities.

**Parameters:**
- p_ttb: [0.1, 0.5]
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
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb_core = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
    # Tallying Prediction
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_tally_core
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
[0] rationale: This metric leverages a mathematical invariant in the Strategy Mixture model. In trials 1 and 2, Tallying predicts a perfect tie, so the difference P(A|T1) - P(A|T2) is driven entirely by Take-The-Best (TTB). In trials 5 and 6, Tallying predicts an identical 3-vs-2 advantage for Option A, meaning the difference P(A|T5) - P(A|T6) is again driven entirely by TTB. Because the Mixture model assumes a linear combination of these strategies, the TTB effect perfectly cancels out, making this double-difference exactly 0.0 for the Mixture model, regardless of any parameter values (variance = 0). For the Sequential Evidence Accumulation (SEA) model, evidence integration is non-linear (softmax on cumulative sums). For high thresholds, the metric evaluates to P(0.5) - P(0.3) - P(0.0) + P(0.1), which is strictly positive due to the monotonic nature of the choice function. Thus, SEA predicts a strictly positive value, while Mixture predicts exactly zero, providing a highly reliable discrimination.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials by Option A's feature vector
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the proportion of choosing Option A (response == 0) for each trial type
    # Trial 1: A=[1, 0, 0, 1, 0]
    t1_mask = data['a_str'] == '10010'
    # Trial 2: A=[0, 1, 1, 0, 0]
    t2_mask = data['a_str'] == '01100'
    # Trial 5: A=[1, 0, 0, 1, 1]
    t5_mask = data['a_str'] == '10011'
    # Trial 6: A=[0, 1, 0, 1, 1]
    t6_mask = data['a_str'] == '01011'
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean() if t1_mask.any() else 0.5
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean() if t2_mask.any() else 0.5
    p_a_t5 = (data.loc[t5_mask, 'response'] == 0).mean() if t5_mask.any() else 0.5
    p_a_t6 = (data.loc[t6_mask, 'response'] == 0).mean() if t6_mask.any() else 0.5
    
    # The metric is a double-difference that isolates the interaction between TTB and Tallying
    return float((p_a_t5 - p_a_t6) - (p_a_t1 - p_a_t2))
outcome: self_sim=0.0062 (var=0.0516) adversary_sim=0.0587 (var=0.0391) welch_t=-0.871 p=0.3879 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    t1_mask = a_str == '10010'\n    t2_mask = a_str == '01100'\n    t3_mask = a_str == '10000'\n    t4_mask = a_str == '01111'\n    \n    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean() if t1_mask.any() else 0.5\n    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean() if t2_mask.any() else 0.5\n    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean() if t3_mask.any() else 0.5\n    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean() if t4_mask.any() else 0.5\n    \n    return float((p_a_t4 - p_a_t2) - (p_a_t3 - p_a_t1))",
  "rationale": "This metric calculates a double difference: (P(A|T4) - P(A|T2)) - (P(A|T3) - P(A|T1)). In the Strategy Mixture model, T4 vs T2 isolates the effect of Tallying when Take-The-Best (TTB) favors Option B. Since Tallying strongly favors A in T4 but ties in T2, this difference is heavily positive (~ 0.5 * (1 - p_ttb)). Similarly, T3 vs T1 isolates Tallying when TTB favors A, yielding a strongly negative difference. Because the Mixture model is constrained to favor Tallying (p_ttb <= 0.5), it predicts a large positive value for this double difference (mean ~0.5). Conversely, for the Sequential Evidence Accumulation (SEA) model, if the threshold is low (terminating at the first cue), P(A|T4) exactly equals P(A|T2) and P(A|T3) exactly equals P(A|T1), reducing the metric to exactly 0.0. Even at higher thresholds, the non-linear evidence integration yields a much smaller positive value. Thus, the SEA model predicts a significantly lower mean (~0.25) with a distribution anchored at zero, reliably discriminating it from the Strategy Mixture model."
}
```

## Usage

```json
{
  "prompt_token_count": 4431,
  "candidates_token_count": 630,
  "total_token_count": 27272
}
```
