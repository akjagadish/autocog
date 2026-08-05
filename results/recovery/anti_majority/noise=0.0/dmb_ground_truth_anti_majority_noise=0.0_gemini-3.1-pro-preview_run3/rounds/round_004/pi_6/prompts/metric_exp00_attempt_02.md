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

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 8: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** This design leverages the lower-cue identity invariance of the Strategy Mixture (TTB + Tallying) model to quantitatively dissociate it from the Weighted Additive with Diminishing Returns (WADD-DR) model. The Mixture model's predictions depend exclusively on the identity of the highest-validity discriminating cue (for TTB) and the total count of winning cues (for Tallying). By holding both the highest-validity cue and the tally difference strictly constant across paired trials (e.g., Trials 1 vs. 2, and Trials 4 vs. 5), the Strategy Mixture model is mathematically forced to predict identical choice probabilities. However, WADD-DR integrates the specific validities of all discriminating cues. By swapping which specific lower-validity cues are won by Option A versus Option B, we alter the accumulated evidence for each option. The concave transformation in WADD-DR then acts on these different evidence totals, leading to systematically shifting choice probabilities across trials where the Mixture model predicts strict equivalence.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive with Diminishing Returns (WADD-DR): Decision-makers integrate all available discriminating cues but apply a sub-additive (concave) transformation to the accumulated evidence. Cue validities are first scaled non-linearly to reflect subjective weighting. Then, the total accumulated evidence for each option undergoes a concave transformation before being converted to choice probabilities. This naturally accounts for the dilution effect when multiple weaker cues are added, pulling choice probabilities toward 0.5, while maintaining sensitivity to the overall balance of evidence.

**Parameters:**
- gamma: [0.1, 30.0]
- alpha: [0.01, 1.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities: subtract 0.5 so a random cue provides 0 evidence, then apply non-linear scaling
    w = np.maximum(0.0, val - 0.5) ** gamma
    
    # Accumulate evidence for each option based on discriminating cues
    diff = a - b
    ev_a = np.sum(w[diff > 0])
    ev_b = np.sum(w[diff < 0])
    
    # Apply sub-additive (concave) transformation to accumulated evidence
    # alpha < 1 yields diminishing returns for additional evidence
    ev_a_trans = (ev_a + 1e-9) ** alpha
    ev_b_trans = (ev_b + 1e-9) ** alpha
    
    # Convert transformed evidence to choice probabilities via softmax
    scores = np.array([ev_a_trans, ev_b_trans])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Strategy Mixture (TTB + Tallying): Decision makers do not universally adhere to a single strategy. Instead, they use a probabilistic mixture of a non-compensatory strategy (Take The Best) and a compensatory strategy (Tallying). A parameter P_TTB dictates the probability of using TTB on any given trial, while 1 - P_TTB is the probability of using Tallying. This accounts for intermediate levels of TTB-consistency and Tallying-consistency observed in empirical data across subjects and trials. The mixture captures a balance between TTB and Tallying, avoiding over-reliance on uniform guessing.

**Parameters:**
- beta: [0.1, 20.0]
- p_ttb: [0.35, 0.95]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.0, 0.0])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    # Normalize by n_features to keep scale comparable to TTB for the shared beta
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta = float(parameters["beta"])
    
    # TTB Probabilities
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally_dist = e_tally / e_tally.sum()
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_core = p_ttb_weight * p_ttb_dist + (1.0 - p_ttb_weight) * p_tally_dist
    
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
[0] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 1 and Trial 2, and between Trial 5 and Trial 4. In the Strategy Mixture model, the highest-validity discriminating cue and the overall tally difference are identical within these pairs, forcing the model to predict exactly the same choice probabilities (difference = 0). For WADD-DR, however, the specific lower-validity cues are swapped between the options, changing the accumulated evidence. Due to the concave transformation applied to the evidence, WADD-DR predicts a systematic shift in choice probabilities between these paired trials, resulting in a non-zero metric value.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    A_str = data['option_a_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    B_str = data['option_b_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    
    chose_A = 1.0 - data['response']
    
    # Trial 1 vs 2
    m1 = (A_str == '11100') & (B_str == '00011')
    m2 = (A_str == '10011') & (B_str == '01100')
    
    # Trial 4 vs 5
    m4 = (A_str == '11001') & (B_str == '10110')
    m5 = (A_str == '11100') & (B_str == '10011')
    
    p1 = chose_A[m1].mean() if m1.any() else 0.5
    p2 = chose_A[m2].mean() if m2.any() else 0.5
    p4 = chose_A[m4].mean() if m4.any() else 0.5
    p5 = chose_A[m5].mean() if m5.any() else 0.5
    
    return float((p1 - p2) + (p5 - p4))
outcome: self_sim=0.1067 (var=0.1020) adversary_sim=-0.0017 (var=0.0362) welch_t=+1.457 p=0.1531 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the difference in the probability of choosing Option A between a 'High WADD-DR' trial set and a 'Low WADD-DR' trial set. The trial sets are carefully constructed so that the Strategy Mixture (TTB + Tallying) model predicts exactly identical choice probabilities for each corresponding pair (Trial 1 matches 2, Trial 5 matches 4, Trial 8 matches 3) because the highest-validity cue and the tally difference are constant. WADD-DR, however, predicts a significantly higher probability of choosing A in the 'High' set because Option A wins on stronger cues overall. By pooling 36 trials per subject for each set, we drastically reduce the per-subject variance compared to a single-trial contrast, yielding a highly stable, significant gap between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    A_str = data['option_a_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    B_str = data['option_b_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    
    chose_A = 1.0 - data['response']
    
    # High WADD-DR advantage trials
    # Trial 1, Trial 5, Trial 8
    high_wadd_mask = (
        ((A_str == '11100') & (B_str == '00011')) | 
        ((A_str == '11100') & (B_str == '10011')) | 
        ((A_str == '10110') & (B_str == '01001'))
    )
    
    # Low WADD-DR advantage trials (perfectly matched to the above in Strategy Mixture)
    # Trial 2, Trial 4, Trial 3
    low_wadd_mask = (
        ((A_str == '10011') & (B_str == '01100')) | 
        ((A_str == '11001') & (B_str == '10110')) | 
        ((A_str == '10101') & (B_str == '01010'))
    )
    
    p_high = chose_A[high_wadd_mask].mean() if high_wadd_mask.any() else 0.5
    p_low = chose_A[low_wadd_mask].mean() if low_wadd_mask.any() else 0.5
    
    return float(p_high - p_low)
outcome: self_sim=0.0372 (var=0.0143) adversary_sim=-0.0006 (var=0.0048) welch_t=+1.368 p=0.1794 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    # Convert rating lists to string keys for safe hashing\n    A_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    B_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # 1.0 if A was chosen, 0.0 if B was chosen\n    chose_A = 1.0 - data['response']\n    \n    # Identify the relevant trial pairs\n    m1 = (A_str == '11100') & (B_str == '00011')\n    m2 = (A_str == '10011') & (B_str == '01100')\n    m3 = (A_str == '10101') & (B_str == '01010')\n    m4 = (A_str == '11001') & (B_str == '10110')\n    m5 = (A_str == '11100') & (B_str == '10011')\n    m8 = (A_str == '10110') & (B_str == '01001')\n    \n    # Calculate empirical probability of choosing A for each trial type\n    p1 = chose_A[m1].mean() if m1.any() else 0.5\n    p2 = chose_A[m2].mean() if m2.any() else 0.5\n    p3 = chose_A[m3].mean() if m3.any() else 0.5\n    p4 = chose_A[m4].mean() if m4.any() else 0.5\n    p5 = chose_A[m5].mean() if m5.any() else 0.5\n    p8 = chose_A[m8].mean() if m8.any() else 0.5\n    \n    # Contrast 1: The 'WADD-DR Advantage'\n    # WADD-DR predicts a strong preference for A in Trial 1 vs Trials 2 & 3 due to accumulated cue validities.\n    # Strategy Mixture predicts exactly ZERO difference because TTB and Tallying scores are identical across these.\n    metric_wadd = p1 - (p2 + p3) / 2.0\n    \n    # Contrast 2: The 'Mixture Advantage'\n    # Mixture predicts a higher preference for A in {2, 3, 8} vs {4, 5} because Tallying favors A in the former but is tied in the latter.\n    # WADD-DR predicts roughly zero difference due to the specific swapped validities and its concave transformation.\n    metric_mix = (p2 + p3 + p8) / 3.0 - (p4 + p5) / 2.0\n    \n    # Final metric subtracts the two to force a qualitative sign reversal between the theories\n    return float(metric_wadd - metric_mix)",
  "rationale": "This metric mathematically isolates the unique behavioral signatures of both theories and pits them against each other to guarantee a qualitative sign reversal. It constructs two orthogonal contrasts. The first ('Metric_WADD') compares Trial 1 against Trials 2 and 3. WADD-DR strongly differentiates these due to integrated cue evidence, while the Strategy Mixture model predicts exactly identical choice probabilities because both the highest-validity cue and the Tallying score are perfectly constant. The second contrast ('Metric_Mix') compares Trials {2,3,8} against {4,5}. Here, the Mixture model expects a large drop in P(A) because Tallying shifts from favoring Option A to a perfect tie, whereas WADD-DR evaluates these sets as roughly equal in accumulated evidence. By subtracting Metric_Mix from Metric_WADD, the final value is guaranteed to be strictly positive for WADD-DR and strictly negative for the Strategy Mixture model across a vast majority of their parameter spaces. Furthermore, by pooling 72 trials per subject across 6 different trial types, the metric heavily suppresses binomial noise, ensuring the between-subject variance remains low enough to easily achieve statistical significance at N=25."
}
```

## Usage

```json
{
  "prompt_token_count": 5031,
  "candidates_token_count": 1021,
  "total_token_count": 27062
}
```
