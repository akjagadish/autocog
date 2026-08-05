# metric_exp01_attempt_00

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
**Validities (n_features=7):** [0.95, 0.95, 0.85, 0.55, 0.55, 0.55, 0.55]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Noisy Weighted Additive (WADD) model from Noisy Tallying, we use a 7-feature environment with three highly valid features and four low-validity features. Tallying ignores validities and counts active features. WADD computes a weighted sum. In the key dissociation trial, an option with three highly valid features is pitted against an option with four low-validity features. Tallying predicts a preference for the latter (4 vs. 3 wins), whereas WADD predicts a preference for the former due to higher summed validities. Additional trials where Tallying predicts a tie but WADD predicts a strict preference, or where both models agree, further isolate the effect of validity weighting.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Noisy Weighted Additive (WADD) Model: Decision-makers compute a global value for each option by summing the validities of its active features. Rather than relying on a single discriminating feature (like Take-The-Best) or ignoring validities (like Tallying), WADD integrates all available validities compensatorily. To account for the highly consistent ~0.5 choice probabilities and near-zero differences observed across the experiments, the decision process incorporates a highly prominent noise mechanism. This is modeled via a very low inverse temperature in the softmax choice rule and a potentially high uniform lapse rate, smoothing out predictions and avoiding deterministic swings.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # WADD: Sum of validities for active features for each option
    scores = np.sum(stim * validities, axis=1)
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Noisy Tallying (Equal Weights): Decision-makers ignore feature validities and instead rely on a simple tallying heuristic, counting the total number of positive features (1s) for each option. They tend to choose the option with the higher count. To account for the highly consistent ~0.5 choice probabilities observed across experiments, this process is heavily smoothed by a prominent noise mechanism, modeled via a high uniform lapse rate and a low inverse temperature in the softmax choice rule.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.5, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying: Sum of active features for each option (ignoring validities)
    scores = np.sum(stim, axis=1)
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
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
[0] rationale: To maximally discriminate WADD from Tallying while minimizing per-subject variance, we compute the proportion of times the subject chose Option A on the three trials where WADD predicts a preference for A but Tallying predicts either a preference for B (Trial 1) or indifference (Trials 2 and 5). By pooling these three trials, we triple the number of observations per subject relative to using just Trial 1, which heavily suppresses between-subject variance and yields a much more stable metric for the Welch's t-test.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_keys = [
        (1, 1, 0, 0, 0, 0),
        (1, 0, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 0)
    ]
    data['a_key'] = data['option_a_ratings'].apply(tuple)
    mask = data['a_key'].isin(a_keys)
    relevant_data = data[mask]
    if len(relevant_data) == 0:
        return 0.5
    return float((relevant_data['response'] == 0).mean())
outcome: self_sim=0.5025 (var=0.0069) adversary_sim=0.5058 (var=0.0038) welch_t=-0.162 p=0.8724 (N=25, alpha=0.01) -> reject

[1] rationale: To effectively discriminate WADD from Tallying despite the very high noise parameters, we calculate the aggregate proportion of choosing Option A across Trials 1 through 5. Trial 6 is excluded because both models strongly predict a preference for A. On the remaining five trials, Tallying predicts either a preference for B or indifference (tally differences of -1, 0, -1, -1, 0), resulting in an expected choice proportion for A strictly below 0.5. In contrast, WADD's validity-weighted differences on these same five trials (+0.2, +0.3, -0.25, -0.2, +0.4) sum to a positive value, yielding an expected choice proportion for A above 0.5. By pooling these five trials, we maximize the directional divergence between the two models while heavily suppressing between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['a_key'] = data['option_a_ratings'].apply(tuple)
    # Exclude Trial 6 where both models strongly prefer A
    mask = data['a_key'] != (1, 1, 1, 0, 0, 0)
    relevant_data = data[mask]
    if len(relevant_data) == 0:
        return 0.5
    # Return the overall proportion of choosing Option A
    return float((relevant_data['response'] == 0).mean())
outcome: self_sim=0.5078 (var=0.0044) adversary_sim=0.4955 (var=0.0027) welch_t=+0.727 p=0.4712 (N=25, alpha=0.01) -> reject

[2] rationale: Under the Noisy Tallying model, Trials 1, 3, and 4 all feature exactly the same tally difference (Option A has 1 fewer positive feature than Option B). Consequently, Tallying predicts identical choice probabilities for Option A across these three trials, making the expected value of the contrast `2 * P(A|T1) - P(A|T3) - P(A|T4)` exactly zero. By contrast, the WADD model predicts a strong preference for Option A in Trial 1 (validity difference +0.20), but a preference for Option B in Trials 3 and 4 (validity differences -0.25 and -0.20). Therefore, WADD predicts a strictly positive and large value for this contrast. By zeroing out the Tallying prediction analytically, we maximally discriminate the two models regardless of noise levels.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['a_key'] = data['option_a_ratings'].apply(tuple)
    
    t1_mask = data['a_key'] == (1, 1, 0, 0, 0, 0)
    t3_mask = data['a_key'] == (0, 1, 1, 0, 0, 0)
    t4_mask = data['a_key'] == (1, 0, 0, 0, 0, 0)
    
    p_t1 = (data[t1_mask]['response'] == 0).mean() if t1_mask.sum() > 0 else 0.5
    p_t3 = (data[t3_mask]['response'] == 0).mean() if t3_mask.sum() > 0 else 0.5
    p_t4 = (data[t4_mask]['response'] == 0).mean() if t4_mask.sum() > 0 else 0.5
    
    return float(2 * p_t1 - p_t3 - p_t4)
outcome: self_sim=-0.0113 (var=0.0693) adversary_sim=-0.0100 (var=0.0715) welch_t=-0.017 p=0.9868 (N=25, alpha=0.01) -> reject

[3] rationale: To maximally discriminate WADD from Tallying while minimizing variance, we isolate the specific trials where the two models make divergent predictions. By computing the WADD and Tally scores for all options dynamically, we can identify trials where the WADD-preferred option has a tally score that is LESS THAN OR EQUAL TO the non-preferred option. This logical filter perfectly isolates Trials 1, 2, and 5, regardless of how the options are randomly assigned to the left/right (A/B) positions on screen (which previous static-key approaches failed to account for, losing half the data). On these trials, WADD predicts a preference for the WADD-preferred option (> 0.5), whereas Tallying predicts either indifference or a preference for the opposite option (< 0.5). Pooling these trials across the full valid dataset maximizes the directional divergence while halving the per-subject variance compared to previous attempts.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    wadd_a = np.sum(a_ratings * validities, axis=1)
    wadd_b = np.sum(b_ratings * validities, axis=1)
    
    tally_a = np.sum(a_ratings, axis=1)
    tally_b = np.sum(b_ratings, axis=1)
    
    wadd_pref = (wadd_b > wadd_a).astype(int)
    
    tally_wadd_pref = np.where(wadd_pref == 0, tally_a, tally_b)
    tally_wadd_nonpref = np.where(wadd_pref == 0, tally_b, tally_a)
    
    mask = tally_wadd_pref <= tally_wadd_nonpref
    
    if not np.any(mask):
        return 0.5
        
    relevant_responses = data['response'].values[mask]
    relevant_wadd_pref = wadd_pref[mask]
    
    chose_wadd_pref = (relevant_responses == relevant_wadd_pref)
    
    return float(np.mean(chose_wadd_pref))
outcome: self_sim=0.5025 (var=0.0069) adversary_sim=0.5058 (var=0.0038) welch_t=-0.162 p=0.8724 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    t1_a = '1110000'\n    t2_a = '1100000'\n    t2_b = '0000110'\n    t4_a = '1001100'\n    \n    mask = (\n        (data['a_str'] == t1_a) |\n        ((data['a_str'] == t2_a) & (data['b_str'] == t2_b)) |\n        (data['a_str'] == t4_a)\n    )\n    \n    relevant = data[mask]\n    if len(relevant) == 0:\n        return 0.5\n        \n    return float((relevant['response'] == 0).mean())",
  "rationale": "To maximally discriminate WADD from Tallying while minimizing variance, we pool Trials 1, 2, and 4. For Tallying, the tally differences (A - B) on these trials are -1, 0, and +1, respectively. Because the softmax function is symmetric, the expected probability of choosing A is P(A|T1) = 1 - P(A|T4), and P(A|T2) = 0.5. Therefore, the aggregate expected proportion of choosing A across these three trials is analytically exactly 0.5 for Tallying, completely invariant to the extreme noise parameters. By contrast, WADD predicts a strict preference for Option A on all three of these trials (validity differences are +0.55, +0.80, and +0.25). Consequently, WADD predicts an aggregate choice proportion strictly greater than 0.5. Pooling these three trials yields 72 observations per subject, which massively suppresses between-subject variance while maintaining a robust, directional mean divergence between the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 4926,
  "candidates_token_count": 515,
  "total_token_count": 9940
}
```
