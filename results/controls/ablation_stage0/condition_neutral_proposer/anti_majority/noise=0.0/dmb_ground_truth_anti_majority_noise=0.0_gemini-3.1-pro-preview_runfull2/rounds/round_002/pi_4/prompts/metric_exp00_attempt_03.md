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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 9: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** To cleanly dissociate the Strategy Mixture (TTB + WADD) from the non-linear WADD model, we must exploit how each theory handles the transition from non-compensatory to compensatory choices. The Mixture model predicts choices as a linear combination of a step-function (TTB) and a linear compensatory function (WADD). In contrast, the non-linear WADD model applies an exponential scaling (gamma) to the log-odds of the validities before summing them, which alters the relative spacing of the cue weights. By systematically varying the number and validity of cues that oppose the single most valid discriminating cue (e.g., pitting Cue 1 against Cues 2+3+4+5, then 2+3+4, then 2+3, etc.), we generate a gradient of compensatory support. The mixture model will produce choice probabilities that reflect a fixed baseline probability of choosing the TTB option (via alpha) plus a softmax of the linear WADD difference. The non-linear WADD model, depending on gamma, will exhibit a fundamentally different psychometric curve across these trade-offs because the marginal contribution of lower-validity cues shrinks or grows non-linearly.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Mixture Theory (TTB + WADD): Decision makers do not universally adopt a single monolithic strategy. Instead, choices are generated from a probabilistic mixture of decision rules. On any given trial, an individual uses a non-compensatory heuristic (Take The Best) with probability 'alpha', and a compensatory strategy (Weighted Additive - WADD) with probability '1 - alpha'. Mixing these strategies captures intermediate rates of compensatory and non-compensatory choices, while WADD leverages cue validities for a more nuanced compensatory evaluation.

**Parameters:**
- alpha: [0.5, 1.0]
- beta: [1.0, 20.0]
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
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Strategy 1: Take The Best (TTB)
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
        z_ttb = beta * (scores_ttb - scores_ttb.max())
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / e_ttb.sum()
        
    # Strategy 2: WADD (Weighted Additive)
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of the two strategies
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Apply lapse rate
    n_opts = p_mix.shape[0]
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
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
**Description:** The Weighted Additive (WADD) model with non-linear weight scaling posits that decision makers integrate all available information by computing a weighted sum of features. However, the weighting of cues is not strictly proportional to their log-odds validity. Instead, decision makers apply a non-linear transformation (parameterized by gamma) to the log-odds, allowing them to stretch the weight differential. This permits WADD to approximate lexicographic (TTB-like) choice when gamma > 1, or more uniform (Tallying-like) weighting when gamma < 1, while remaining a fully compensatory integration process.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- gamma: [0.5, 2.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds weights, clipping to avoid infinity
    val_clipped = np.clip(val, 0.5001, 0.9999)
    log_odds = np.log(val_clipped / (1.0 - val_clipped))
    
    gamma = float(parameters["gamma"])
    w = np.sign(log_odds) * (np.abs(log_odds) ** gamma)
    
    # Compute weighted sum for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the weighted scores
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
[0] rationale: This metric isolates the transition point where the compensatory strategy (WADD) flips its preference to align with the non-compensatory strategy (TTB). By pitting the most valid cue against a decreasing number of opposing cues, we trace a psychometric curve. Strategy Mixture evaluates WADD linearly based on raw validities, meaning WADD still opposes TTB in the '01100' trial but aligns with it in the '01000' trial, creating a sharp jump in choice probability between these two. Non-linear WADD transforms validities into log-odds, making the single best cue strong enough to outweigh '01100' but not '01110', shifting the jump to occur between '01110' and '01100'. The discrete second derivative captures this phase shift effectively, yielding opposite signs for the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    def get_p_target(df, target_str, opp_str):
        mask1 = (df['A_str'] == target_str) & (df['B_str'] == opp_str)
        mask2 = (df['A_str'] == opp_str) & (df['B_str'] == target_str)
        
        chose_target = 0
        total = 0
        
        if mask1.sum() > 0:
            chose_target += (df.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_target += (df.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
            
        if total == 0:
            return 0.5
        return chose_target / total

    # Evaluate the probability of choosing the TTB-favored option [1,0,0,0,0]
    # against progressively weaker WADD-favored opponents.
    p2 = get_p_target(data, '10000', '01110')
    p3 = get_p_target(data, '10000', '01100')
    p4 = get_p_target(data, '10000', '01000')
    
    # We calculate the discrete second derivative (curvature) of the choice probability.
    # Strategy Mixture jumps between p3 and p4, whereas Non-Linear WADD jumps between p2 and p3.
    return float(p4 - 2 * p3 + p2)

outcome: self_sim=0.1844 (var=0.0704) adversary_sim=0.0711 (var=0.1688) welch_t=+1.159 p=0.2533 (N=25, alpha=0.01) -> reject

[1] rationale: In all trials, the non-compensatory heuristic (Take The Best) strictly favors Option A. The Strategy Mixture model assumes choices are a mixture of TTB (with probability alpha >= 0.5) and a linear WADD model. Because TTB always chooses A, the choice probability for A in the Mixture model has a hard floor at 'alpha'. Even when Option B is overwhelmingly strong in linear WADD (like in Trial 1), the Mixture model still predicts P(A) ~ alpha. When B is weak (Trial 4), P(A) rises slightly to alpha + (1-alpha)*P_WADD. Thus, the difference P(A|T4) - P(A|T1) is tightly constrained and small for the Mixture model.

In contrast, the Non-linear WADD model evaluates choices using a pure softmax over non-linearly transformed weights. If the non-linear scaling parameter (gamma) is small, Option B easily overwhelms Option A in Trial 1, pushing P(A|T1) near 0, while P(A|T4) remains high. This creates a massive drop in choice probability between the two trials, leading to a much larger difference on average compared to the Mixture model's constrained step.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Extract choices for specific trials to compare the drop in choice probability
    # when B gains an additional opposing cue.
    # Trial 4: A=[1,0,0,0,0] vs B=[0,1,0,0,0]
    # Trial 1: A=[1,0,0,0,0] vs B=[0,1,1,1,1]
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    def get_p_a(df, a_str, b_str):
        mask1 = (df['A_str'] == a_str) & (df['B_str'] == b_str)
        mask2 = (df['A_str'] == b_str) & (df['B_str'] == a_str)
        
        chose_a = 0
        total = 0
        
        if mask1.sum() > 0:
            chose_a += (df.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_a += (df.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
            
        if total == 0:
            return 0.5
        return chose_a / total

    p_a_t4 = get_p_a(data, '10000', '01000')
    p_a_t1 = get_p_a(data, '10000', '01111')
    
    return float(p_a_t4 - p_a_t1)
outcome: self_sim=0.1778 (var=0.0440) adversary_sim=0.2444 (var=0.1175) welch_t=-0.830 p=0.4118 (N=25, alpha=0.01) -> reject

[2] rationale: We identify 7 specific trial types where the non-compensatory heuristic (TTB) always favors Option A, but the linear compensatory strategy (WADD on raw validities) always favors Option B. According to the Strategy Mixture theory, the probability of choosing A in every single one of these trials is firmly pinned to the mixture parameter 'alpha', because the two sub-strategies never change their minds across this set. Thus, the variance of P(A) across these 7 trials should be approximately zero. In contrast, the Non-linear WADD theory evaluates these trials using a log-odds transformation, which causes its preference to shift dynamically from strongly preferring B to strongly preferring A across the set. This results in a much higher variance of P(A) across these specific trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # These 7 trial types have a special property:
    # 1. Take-The-Best (TTB) strictly prefers Option A in all of them.
    # 2. Linear WADD (using raw validities) strictly prefers Option B in all of them.
    # Therefore, the Strategy Mixture model predicts P(A) = alpha for ALL of these trials,
    # meaning the variance of P(A) across these 7 trials should be near 0.
    # In contrast, the Non-linear WADD model evaluates them using transformed log-odds,
    # causing its preference to flip between A and B across these trials, yielding high variance.
    target_pairs = [
        ('10000', '01111'),
        ('10000', '01110'),
        ('10000', '01100'),
        ('01000', '00111'),
        ('01000', '00110'),
        ('11000', '10111'),
        ('10100', '01011')
    ]
    
    p_a_list = []
    for a_str, b_str in target_pairs:
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        
        chose_a = 0
        total = 0
        
        if mask1.sum() > 0:
            chose_a += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_a += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
            
        if total > 0:
            p_a_list.append(chose_a / total)
            
    if len(p_a_list) < 2:
        return 0.0
        
    return float(np.var(p_a_list, ddof=0))
outcome: self_sim=0.0008 (var=0.0001) adversary_sim=0.0055 (var=0.0009) welch_t=-0.727 p=0.4729 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    # Trials where Non-Linear WADD strongly prefers Option A (due to log-odds scaling)\n    high_pairs = [\n        ('10100', '01011'), # Trial 10\n        ('11000', '10110'), # Trial 9\n        ('10000', '01000')  # Trial 4\n    ]\n    \n    # Trials where Non-Linear WADD strongly prefers Option B (for gamma < 1.5)\n    low_pairs = [\n        ('10000', '01111'), # Trial 1\n        ('10000', '01110'), # Trial 2\n        ('01000', '00111')  # Trial 5\n    ]\n    \n    def get_p_a(pairs):\n        chose_a = 0\n        total = 0\n        for a_str, b_str in pairs:\n            mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)\n            mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)\n            \n            if mask1.any():\n                chose_a += (data.loc[mask1, 'response'] == 0).sum()\n                total += mask1.sum()\n            if mask2.any():\n                chose_a += (data.loc[mask2, 'response'] == 1).sum()\n                total += mask2.sum()\n        if total == 0:\n            return 0.5\n        return chose_a / total\n\n    p_high = get_p_a(high_pairs)\n    p_low = get_p_a(low_pairs)\n    \n    return float(p_high - p_low)",
  "rationale": "By aggregating trials into two distinct sets ('high' and 'low'), we can exploit the non-linear log-odds transformation of the NL-WADD model. In the 'high' group, NL-WADD strongly prefers Option A because the top validities dominate. In the 'low' group, NL-WADD frequently prefers Option B because the sum of opposing lower validities outweighs the top ones for typical gamma values. Thus, NL-WADD predicts a massive difference in P(A) between these groups. Conversely, the Strategy Mixture Theory (SMT) is anchored by the Take-The-Best heuristic, which strictly prefers Option A across ALL of these trials. This forces SMT's P(A) to be bounded below by the mixture parameter 'alpha' (>= 0.5) in both groups. Because the linear WADD component of SMT only weakly modulates this probability, SMT predicts a very small difference between the two groups. Averaging over 3 trials per group reduces binomial sampling noise, ensuring tight between-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 6240,
  "candidates_token_count": 783,
  "total_token_count": 37270
}
```
