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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate Generalized WADD from Strategy Mixture Theory, we manipulate the margin by which the compensatory WADD strategy opposes the non-compensatory TTB strategy. Strategy Mixture Theory assumes choices are a stable probabilistic coin-flip between pure TTB and pure WADD. Therefore, on any trial where TTB strongly favors Option A and WADD strongly favors Option B, Strategy Mixture predicts a relatively constant intermediate choice probability driven by the mixing parameter 'alpha'. In contrast, Generalized WADD integrates all features non-linearly. A single 'gamma' parameter cannot flatten the varying evidence margins across different trials. By presenting a spectrum of conflicts—ranging from the top cue being opposed by all four remaining cues, to being opposed by only two—Generalized WADD is forced to predict a graded shift in probabilities, whereas Strategy Mixture predicts a step-function or constant mixture across these conflicts.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People make decisions by integrating all available information rather than stopping at the first discriminating cue or simply counting features. According to the Weighted Additive (WADD) theory, decision-makers evaluate each option by computing a sum of its features, weighted by the subjective validity of each feature. However, people may non-linearly amplify the importance of highly valid cues. To capture this, validities are exponentiated by a scaling parameter and then normalized, allowing the model to smoothly interpolate between highly compensatory (Tallying-like) and non-compensatory (TTB-like) decision strategies without shrinking the overall scale of the evidence. Choice probabilities are generated via a softmax function over the weighted sums, with an independent lapse rate for random guessing.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [1.0, 30.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = val ** gamma
    val = val / np.sum(val)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Strategy Mixture Theory: Decision-makers do not uniformly rely on a single decision strategy. Instead, they exhibit variability in strategy selection, probabilistically mixing between a non-compensatory heuristic (Take The Best) and a compensatory strategy (Weighted Additive, WADD). The mixture parameter 'alpha' dictates the probability of using TTB on any given trial, while '1 - alpha' is the probability of using WADD. This allows the model to capture intermediate choice proportions that neither pure strategy can explain on its own, with a bias toward the empirically dominant non-compensatory strategy.

**Parameters:**
- alpha: [0.5, 1.0]
- beta: [1.0, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    beta = float(parameters["beta"])
    
    # Take The Best (TTB) prediction
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
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # WADD prediction
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores_wadd = np.array([score_a, score_b])
    
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Strategy Mixture
    alpha = float(parameters["alpha"])
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
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
[0] rationale: Strategy Mixture Theory assumes choices on conflict trials are a probabilistic coin-flip between pure TTB and pure WADD. In Trials 1, 3, 4, and 6, TTB always favors Option A, and pure WADD always favors Option B, so Strategy Mixture predicts a flat choice probability P(A) = alpha across these trials. Generalized WADD, however, non-linearly integrates all cues. It predicts that decreasing the number of cues opposing the top cue (e.g., from 4 opposing cues in Trial 1 to 2 opposing cues in Trial 3) will substantially increase the probability of choosing Option A. This metric computes the average shift in choice probability when the number of opposing cues drops, which should be roughly zero for Strategy Mixture and distinctly positive for Generalized WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 1, 1, 1))
    t3_mask = (data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 1, 0, 0))
    t4_mask = (data['A_tuple'] == (0, 1, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 1, 1, 1))
    t6_mask = (data['A_tuple'] == (0, 1, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 1, 1, 0))
    
    p_A_t1 = 1.0 - data[t1_mask]['response'].mean()
    p_A_t3 = 1.0 - data[t3_mask]['response'].mean()
    p_A_t4 = 1.0 - data[t4_mask]['response'].mean()
    p_A_t6 = 1.0 - data[t6_mask]['response'].mean()
    
    diff1 = p_A_t3 - p_A_t1
    diff2 = p_A_t6 - p_A_t4
    
    return float((diff1 + diff2) / 2.0)
outcome: self_sim=0.0200 (var=0.0166) adversary_sim=0.0044 (var=0.0167) welch_t=+0.428 p=0.6705 (N=25, alpha=0.01) -> reject

[1] rationale: Strategy Mixture Theory assumes choices are a probabilistic coin-flip between pure TTB and pure WADD. In Trials 1 and 4, TTB favors A and WADD strongly favors B, yielding a choice probability for A of exactly `alpha`. In Trial 5, TTB favors A but WADD predicts a perfect tie, making the choice probability for A `0.5 + 0.5 * alpha`. The difference between Trial 5 and Trials 1/4 is analytically constrained to `0.5 - 0.5 * alpha` for Strategy Mixture (mean ~0.125). In contrast, Generalized WADD integrates all features non-linearly. For a wide range of its `gamma` parameter, it strongly favors Option A in Trial 5 while still strongly favoring Option B in Trials 1 and 4, producing a significantly larger positive difference. By contrasting these specific trials, we maximize the divergence between the theories while keeping the expected variance low.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Create string representations of the ratings to identify trial types
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # We are interested in the proportion of times Option A was chosen
    data['chose_A'] = 1 - data['response']
    
    # Trial 5: WADD predicts a tie (linear sum A=1.4, B=1.4), TTB favors A.
    t5_mask = (data['A_str'] == '010001') | ((data['A_str'] == '01001') & (data['B_str'] == '00110'))
    
    # Trial 1 & Trial 4: WADD strongly favors B, TTB favors A.
    t1_mask = (data['A_str'] == '10000') & (data['B_str'] == '01111')
    t4_mask = (data['A_str'] == '01000') & (data['B_str'] == '00111')
    
    # Compute choice proportions for Option A
    p_a_t5 = data[t5_mask]['chose_A'].mean() if t5_mask.sum() > 0 else 0.5
    p_a_t1 = data[t1_mask]['chose_A'].mean() if t1_mask.sum() > 0 else 0.5
    p_a_t4 = data[t4_mask]['chose_A'].mean() if t4_mask.sum() > 0 else 0.5
    
    # Metric: Difference between Trial 5 and the average of Trials 1 & 4
    return float(p_a_t5 - (p_a_t1 + p_a_t4) / 2.0)
outcome: self_sim=-0.0344 (var=0.0456) adversary_sim=0.1000 (var=0.0227) welch_t=-2.571 p=0.01367 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    data['chose_A'] = 1 - data['response']\n    \n    t5_mask = (data['A_str'] == '01001') & (data['B_str'] == '00110')\n    t6_mask = (data['A_str'] == '01000') & (data['B_str'] == '00110')\n    \n    p_a_t5 = data[t5_mask]['chose_A'].mean() if t5_mask.sum() > 0 else 0.5\n    p_a_t6 = data[t6_mask]['chose_A'].mean() if t6_mask.sum() > 0 else 0.5\n    \n    return float(p_a_t5 - p_a_t6)",
  "rationale": "Strategy Mixture Theory assumes choices are a combination of pure TTB and pure WADD. In Trial 5, TTB favors A and WADD predicts a perfect tie (A=1.4, B=1.4), so Strategy Mixture predicts P(A) = alpha + (1-alpha)*0.5 = 0.5 + 0.5*alpha. In Trial 6, TTB favors A but WADD favors B (A=0.85, B=1.4), so Strategy Mixture predicts P(A) = alpha. The difference P(A|T5) - P(A|T6) is analytically constrained to exactly 0.5 - 0.5*alpha for Strategy Mixture, which falls strictly between 0.0 and 0.25 (mean ~0.125). Conversely, Generalized WADD integrates features non-linearly. For most values of its gamma parameter > 1, the exponentiation causes the single strong cue of Option A in Trial 5 to outweigh the two weaker cues of Option B, leading to a strong preference for A (P(A) near 1). But in Trial 6, dropping the fifth cue often flips the preference completely to B (P(A) near 0). Thus, Generalized WADD predicts a massive difference between these two trials (often > 0.6), cleanly discriminating the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 4885,
  "candidates_token_count": 601,
  "total_token_count": 7479
}
```
