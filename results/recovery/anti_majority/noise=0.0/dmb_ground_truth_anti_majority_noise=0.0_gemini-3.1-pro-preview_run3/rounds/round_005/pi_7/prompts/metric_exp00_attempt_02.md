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

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** This design leverages an invariance property of the Strategy Mixture (TTB + Tallying) model to quantitatively dissociate it from the Sequential Evidence Accumulation model. In the Strategy Mixture model, choice probabilities depend solely on the identity of the highest-validity discriminating cue (for TTB) and the difference in the total count of winning cues (for Tallying). By holding the highest-validity discriminating cue and the total cue-win counts constant across a subset of trials (e.g., Trials 1-3, and Trials 4-6), the Strategy Mixture model is mathematically forced to predict the exact same choice probabilities within each subset. However, the Sequential Evidence Accumulation model integrates the actual validities of the specific lower-tier cues sequentially until a threshold is reached. By shifting which specific lower-validity cues Option B wins across these trials, the accumulated evidence varies. Consequently, if the threshold is large enough to evaluate these cues, the Sequential model predicts systematically shifting choice probabilities across trials where the Strategy Mixture model predicts strict equivalence.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Evidence Accumulation with Flexible Threshold: Decision-makers search through cues sequentially in order of their validity. As they evaluate each cue, they accumulate evidence for the favored option. The search stops as soon as the absolute difference in accumulated evidence between the two options reaches a subjective threshold. If the threshold is low (or zero), this mechanism perfectly mimics Take-The-Best by stopping at the first discriminating cue. If the threshold is high, it evaluates all available cues, naturally transitioning into compensatory strategies like Weighted Additive (WADD) or Tallying.

**Parameters:**
- gamma: [0.0, 3.0]
- theta: [0.0, 1.0]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Search through cues in order of validity (highest first)
    cue_order = np.argsort(-val, kind="stable")
    
    # Scale validities non-linearly to represent subjective evidence weights
    w = val ** gamma
    
    ev_a = 0.0
    ev_b = 0.0
    
    # Sequential evidence accumulation
    for i in cue_order:
        if a[i] > b[i]:
            ev_a += w[i]
        elif b[i] > a[i]:
            ev_b += w[i]
            
        # Stop search if the evidence difference reaches the threshold
        # (and ensure we don't stop prematurely if no discriminating evidence has been found yet)
        if abs(ev_a - ev_b) >= theta and abs(ev_a - ev_b) > 0:
            break
            
    # Convert accumulated evidence into choice probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate (guessing)
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
[0] rationale: This metric isolates subsets of trials where the Strategy Mixture model is mathematically forced to predict identical choice probabilities, because both the highest validity discriminating cue (TTB) and the total cue-win counts (Tallying) are held constant. Specifically, Trials 1 and 3 share identical TTB and Tallying predictions, as do Trials 4 and 6. Therefore, the Strategy Mixture model predicts zero difference in the choice rate of A within these pairs. In contrast, the Sequential Evidence Accumulation model processes cues sequentially; by shifting B's winning cues to lower validities (later in the sequence), the accumulated evidence shifts systematically, predicting a higher choice rate for A in Trial 3 than Trial 1, and in Trial 4 than Trial 6. The sum of these differences will be strictly zero under Strategy Mixture but positive under Sequential Evidence Accumulation.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert rating lists to strings for easy matching
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # Trial 1: A='11000', B='10110'
    # Trial 3: A='11000', B='10011'
    t1_mask = (data['A_str'] == '11000') & (data['B_str'] == '10110')
    t3_mask = (data['A_str'] == '11000') & (data['B_str'] == '10011')
    
    # Trial 4: A='01100', B='00011'
    # Trial 6: A='01001', B='00110'
    t4_mask = (data['A_str'] == '01100') & (data['B_str'] == '00011')
    t6_mask = (data['A_str'] == '01001') & (data['B_str'] == '00110')
    
    # Calculate P(A) = 1 - mean(response)
    p_a_t1 = 1.0 - data.loc[t1_mask, 'response'].mean() if t1_mask.sum() > 0 else 0.5
    p_a_t3 = 1.0 - data.loc[t3_mask, 'response'].mean() if t3_mask.sum() > 0 else 0.5
    p_a_t4 = 1.0 - data.loc[t4_mask, 'response'].mean() if t4_mask.sum() > 0 else 0.5
    p_a_t6 = 1.0 - data.loc[t6_mask, 'response'].mean() if t6_mask.sum() > 0 else 0.5
    
    # Strategy Mixture predicts P(A|T1) == P(A|T3) and P(A|T4) == P(A|T6)
    # Sequential Evidence Accumulation predicts P(A|T3) > P(A|T1) and P(A|T4) > P(A|T6)
    return float((p_a_t3 - p_a_t1) + (p_a_t4 - p_a_t6))
outcome: self_sim=0.0644 (var=0.1265) adversary_sim=-0.0067 (var=0.0703) welch_t=+0.802 p=0.4271 (N=25, alpha=0.01) -> reject

[1] rationale: This metric contrasts trials where Tallying strongly favors the TTB-favored option (Trials 9, 10) against trials where Tallying strongly opposes it (Trials 1, 2, 3, 7). In the Strategy Mixture model, the Tallying component universally pulls choices away from the TTB prediction on the negative trials and reinforces it on the positive trials, resulting in a large, robust positive difference across all parameters. Conversely, the Sequential Evidence Accumulation (SEA) model will stop at the very first cue for the vast majority of subjects (whenever the threshold is low), perfectly mimicking TTB and yielding an expected difference of strictly 0.0. Only the small subset of SEA subjects with high thresholds will show a positive difference, making the overall mean difference for SEA drastically lower than for Strategy Mixture.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert ratings to strings for easy matching
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # Group Positive Tallying: Trials 9, 10
    # t9: A='11100', B='00011'
    # t10: A='11010', B='00101'
    mask_pos = (
        ((data['A_str'] == '11100') & (data['B_str'] == '00011')) |
        ((data['A_str'] == '11010') & (data['B_str'] == '00101'))
    )
    
    # Group Negative Tallying: Trials 1, 2, 3, 7
    # t1: A='11000', B='10110'
    # t2: A='11000', B='10101'
    # t3: A='11000', B='10011'
    # t7: A='10000', B='01111'
    mask_neg = (
        ((data['A_str'] == '11000') & (data['B_str'] == '10110')) |
        ((data['A_str'] == '11000') & (data['B_str'] == '10101')) |
        ((data['A_str'] == '11000') & (data['B_str'] == '10011')) |
        ((data['A_str'] == '10000') & (data['B_str'] == '01111'))
    )
    
    # Calculate choice rate for Option A in both groups
    p_pos = 1.0 - data.loc[mask_pos, 'response'].mean() if mask_pos.sum() > 0 else 0.5
    p_neg = 1.0 - data.loc[mask_neg, 'response'].mean() if mask_neg.sum() > 0 else 0.5
    
    return float(p_pos - p_neg)
outcome: self_sim=0.0867 (var=0.0344) adversary_sim=0.2172 (var=0.0359) welch_t=-2.462 p=0.01748 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Convert rating lists to strings for exact matching\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))\n    \n    # Trial 7: TTB favors A (Cue 1), Tally strongly favors B (4 vs 1)\n    t7_mask = (data['A_str'] == '10000') & (data['B_str'] == '01111')\n    # Trial 8: TTB favors B (Cue 1), Tally strongly favors A (3 vs 1)\n    t8_mask = (data['A_str'] == '00111') & (data['B_str'] == '10000')\n    \n    # Calculate probability of choosing the COMPENSATORY (Tallying) option\n    # In T7, compensatory option is B (response == 1)\n    p_comp_t7 = data.loc[t7_mask, 'response'].mean() if t7_mask.sum() > 0 else 0.5\n    \n    # In T8, compensatory option is A (response == 0)\n    p_comp_t8 = 1.0 - data.loc[t8_mask, 'response'].mean() if t8_mask.sum() > 0 else 0.5\n    \n    # Return the combined rate of compensatory choices on these extreme trials\n    return float(p_comp_t7 + p_comp_t8)",
  "rationale": "This metric measures the rate of compensatory choices on the two trials (T7 and T8) where the non-compensatory (TTB) and compensatory (Tallying/WADD) strategies most strongly disagree. In both trials, the TTB winner is determined by the highest possible validity cue (Cue 1, validity 0.95). For the Sequential Evidence Accumulation (SEA) model to choose the compensatory option, its evidence threshold must be high enough to survive the massive initial evidence from Cue 1. Because the threshold parameter theta is bounded [0.0, 1.0], this rarely happens; SEA almost always stops at Cue 1, effectively mimicking TTB and yielding a very low compensatory choice rate (mostly driven by lapse/guessing). In contrast, the Strategy Mixture model explicitly assumes a probabilistic mixture of TTB and Tallying on every trial. Because the Tallying component strongly favors the compensatory option in these specific trials, the Strategy Mixture model reliably predicts a much higher baseline rate of compensatory choices, leading to a large and robust mean difference between the theories."
}
```

## Usage

```json
{
  "prompt_token_count": 5645,
  "candidates_token_count": 657,
  "total_token_count": 37759
}
```
