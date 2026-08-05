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
**Validities (n_features=5):** [0.98, 0.85, 0.8, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To cleanly dissociate Rank-based Weighting Theory from the Strategy Mixture Theory, we use a validity profile with an extreme outlier at the top (0.98) and a steep drop-off, followed by moderately high validities. The Strategy Mixture Theory includes WADD (which uses log-odds, placing massive weight on the 0.98 cue, effectively dwarfing the others) and Tallying (which ignores weights entirely). Rank-based Weighting, however, strictly maps the validities to an ordinal decay (e.g., 1, 1/2, 1/3...). By pitting the highest-validity cue against the combination of the 2nd, 3rd, 4th, and 5th cues, Rank-based Weighting allows the accumulation of lower-ranked cues to surpass the top cue (since 1/2 + 1/3 + 1/4 + 1/5 > 1). In contrast, pure WADD strongly prefers the single 0.98 cue (log-odds ~3.89 vs sum of others ~3.71). By combining this with trials that decouple TTB and Tallying, we create a signature of choice probabilities that cannot be reconciled by any mixture of WADD, TTB, and Tallying.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Rank-based Weighting Theory posits that decision-makers do not use complex mathematical transformations like log-odds to weigh evidence. Instead, they rely on the simple ordinal ranking of cue validities. Cues are weighted according to an inverse function of their rank (proportional to 1/rank^rho). By restricting the decay parameter rho to a moderate range, the theory maintains a highly compensatory mechanism that prevents over-reliance on the most valid cue, capturing the softer probability matching observed in human multi-attribute decision making.

**Parameters:**
- beta: [0.01, 15.0]
- epsilon: [0.0, 0.5]
- rho: [0.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute ranks (1 is the highest validity)
    # Using stable sort for consistent tie-breaking if validities are equal
    order = np.argsort(-val, kind='stable')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Apply rank-based weighting
    rho = float(parameters["rho"])
    w = 1.0 / (ranks ** rho)
    
    # Compute weighted sum of features for each option
    scores = np.dot(stim, w)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Strategy Mixture Theory with Independent Scaling: Individuals use a probabilistic mixture of distinct heuristics (WADD, Tallying, and Take-The-Best), but because the internal evidence scales of these heuristics vary dramatically (log-odds sums vs. integer counts vs. binary indicators), each heuristic applies its own independent temperature parameter to properly calibrate its choice probabilities before mixing.

**Parameters:**
- beta_wadd: [0.1, 10.0]
- beta_tally: [0.1, 10.0]
- beta_ttb: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- w_wadd: [0.0, 1.0]
- w_tally: [0.0, 1.0]
- w_ttb: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    scores_wadd = np.dot(stim, w)
    
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
    if winner is None:
        scores_ttb = np.array([0.0, 0.0])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
    beta_wadd = float(parameters["beta_wadd"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    
    def get_probs(scores, beta):
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        return e / np.sum(e)
        
    p_wadd = get_probs(scores_wadd, beta_wadd)
    p_tally = get_probs(scores_tally, beta_tally)
    p_ttb = get_probs(scores_ttb, beta_ttb)
    
    w1 = float(parameters["w_wadd"])
    w2 = float(parameters["w_tally"])
    w3 = float(parameters["w_ttb"])
    w_sum = w1 + w2 + w3 + 1e-9
    
    p_mix = (w1 * p_wadd + w2 * p_tally + w3 * p_ttb) / w_sum
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
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
[0] rationale: This metric calculates the proportion of times subjects choose Option A on Trial 3, where Option A has the top two cues (1, 1, 0, 0, 0) and Option B has the bottom three cues (0, 0, 1, 1, 1). According to Rank-based Weighting Theory, the steep decay of weights based on rank strongly favors Option A. However, the Strategy Mixture Theory (incorporating WADD and Tallying) strongly favors Option B, because the sum of log-odds for the three lower cues exceeds the top two, and Tallying simply counts 3 vs 2. This creates a sharp qualitative divergence on this specific trial.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask = (a_str == '11000') & (b_str == '00111')
    if mask.sum() == 0:
        return 0.5
    return float((data.loc[mask, 'response'] == 0).mean())
outcome: self_sim=0.5650 (var=0.0844) adversary_sim=0.4125 (var=0.0294) welch_t=+2.261 p=0.02944 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the proportion of times subjects choose Option A when it is supported only by the highest-validity cue (Trials 1 and 2). Under the Strategy Mixture Theory, WADD and Tallying strongly favor Option B (which combines multiple lower-validity cues), causing a strong overall preference for B. Rank-based Weighting, however, applies a steep rank-based decay, allowing the single most valid cue to outweigh combinations of lower-ranked cues, leading to a much higher rate of choosing Option A. By pooling across Trials 1 and 2, we double the number of observations per subject compared to a single trial, significantly reducing within-subject variance and ensuring a tighter estimate.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    mask = (a_str == '10000')
    if mask.sum() == 0:
        return 0.5
    return float((data.loc[mask, 'response'] == 0).mean())
outcome: self_sim=0.4125 (var=0.0775) adversary_sim=0.3794 (var=0.0204) welch_t=+0.529 p=0.5999 (N=25, alpha=0.01) -> reject

[2] rationale: We examine the difference in the rate of choosing Option A between Trial 6 (A=[1,0,1,0,0], B=[0,1,0,1,0]) and Trial 4 (A=[0,1,1,0,0], B=[0,0,0,1,1]). 

Under Strategy Mixture Theory, WADD evaluates the log-odds difference. Because the validities are tightly clustered, the difference in log-odds between cue 2 and 4 + cue 3 and 5 (Trial 4) is larger (+0.50) than the difference between cue 1 and 2 + cue 3 and 4 (Trial 6, +0.27). Since TTB favors A in both and Tallying ties in both, Strategy Mixture universally predicts a stronger preference for A in Trial 4 than in Trial 6 (i.e., Trial 6 - Trial 4 is negative).

Under Rank-based Weighting Theory, the steep convex decay (1/rank^rho) means the gap between rank 1 and 2 is much larger than the gap between rank 2 and 4. This causes the evidence advantage for Option A to be strictly larger in Trial 6 than in Trial 4 for any rho > 0. Thus, Rank-based Weighting predicts a positive difference. This creates a clean qualitative dissociation (positive vs. negative difference) between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask_t6 = (a_str == '10100')
    mask_t4 = (a_str == '01100')
    
    if mask_t6.sum() == 0 or mask_t4.sum() == 0:
        return 0.0
        
    p_a_t6 = (data.loc[mask_t6, 'response'] == 0).mean()
    p_a_t4 = (data.loc[mask_t4, 'response'] == 0).mean()
    
    return float(p_a_t6 - p_a_t4)
outcome: self_sim=0.0100 (var=0.0252) adversary_sim=-0.0300 (var=0.0280) welch_t=+0.867 p=0.3903 (N=25, alpha=0.01) -> reject

[3] rationale: We contrast the rate of choosing Option A in Trial 1 (A=[1,0,0,0,0], B=[0,1,1,0,0]) versus Trial 5 (A=[0,1,0,0,0], B=[0,0,1,1,0]). In both trials, Option A has a single higher-validity cue, while Option B has two lower-validity cues. Under Strategy Mixture Theory, WADD evaluates the log-odds difference: the combination of cues 2 and 3 outweighs cue 1 by a larger margin than cues 3 and 4 outweigh cue 2. Tallying and TTB yield identical differences across the two trials. Thus, Strategy Mixture predicts subjects will choose Option A slightly less often (or equally) in Trial 1 compared to Trial 5. In contrast, Rank-based Weighting Theory enforces a steep decay based on rank (1/rank^rho). The drop from rank 1 to ranks 2 and 3 is much steeper than the drop from rank 2 to ranks 3 and 4. Consequently, Rank-based Weighting strongly predicts a higher rate of choosing Option A in Trial 1 than in Trial 5. This difference (Trial 1 - Trial 5) cleanly dissociates the two models (positive for Rank-based Weighting, negative/zero for Strategy Mixture).
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask_t1 = (a_str == '10000') & (data['option_b_ratings'].apply(lambda x: ''.join(map(str, x))) == '01100')
    mask_t5 = (a_str == '01000') & (data['option_b_ratings'].apply(lambda x: ''.join(map(str, x))) == '00110')
    
    if mask_t1.sum() == 0 or mask_t5.sum() == 0:
        return 0.0
        
    p_a_t1 = (data.loc[mask_t1, 'response'] == 0).mean()
    p_a_t5 = (data.loc[mask_t5, 'response'] == 0).mean()
    
    return float(p_a_t1 - p_a_t5)
outcome: self_sim=0.1112 (var=0.0390) adversary_sim=0.0050 (var=0.0356) welch_t=+1.946 p=0.05757 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    # Trial 3: A=[0,1,1,0,0], B=[1,0,0,0,1]\n    mask_t3 = a_str == '01100'\n    # Trial 5: A=[1,0,0,1,0], B=[0,1,1,0,1]\n    mask_t5 = a_str == '10010'\n    \n    if mask_t3.sum() == 0 or mask_t5.sum() == 0:\n        return 0.5\n        \n    # We measure the rate of choosing the option that possesses the highest-validity cue (cue 1).\n    # In Trial 3, cue 1 is in Option B. In Trial 5, cue 1 is in Option A.\n    p_cue1_t3 = (data.loc[mask_t3, 'response'] == 1).mean()\n    p_cue1_t5 = (data.loc[mask_t5, 'response'] == 0).mean()\n    \n    return float((p_cue1_t3 + p_cue1_t5) / 2.0)",
  "rationale": "This metric calculates the pooled probability of choosing the option containing the top validity cue across Trials 3 and 5. In both of these trials, the WADD heuristic (part of the Strategy Mixture Theory) calculates an identical, massive log-odds advantage (~0.97) for the option with the top cue, and TTB also universally favors it. Tallying only slightly dampens this (neutral in T3, -1 in T5). Consequently, the Strategy Mixture Theory strongly and consistently predicts a high rate of choosing the top-cue option across these trials. In contrast, Rank-based Weighting Theory employs a steep rank-based decay (1/rank^rho). For low-to-moderate values of rho, the combined weight of the opposing lower-ranked cues frequently matches or exceeds the top cue (e.g., at rho=0 the top-cue option is actually at a disadvantage in T5 and tied in T3). Thus, Rank-based Weighting predicts a significantly lower pooled probability of choosing the top-cue option. By averaging across 32 trials per subject, we sharply reduce within-subject variance, exposing a clean, highly significant divergence between the theories."
}
```

## Usage

```json
{
  "prompt_token_count": 5606,
  "candidates_token_count": 582,
  "total_token_count": 21151
}
```
