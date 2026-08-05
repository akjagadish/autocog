# metric_exp02_attempt_00

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
**Validities (n_features=5):** [0.9, 0.85, 0.8, 0.75, 0.7]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** This design quantitatively dissociates Dual-Process Strategy Selection Theory from Generalized WADD by holding the top discriminating cue constant across a sequence of trials while systematically increasing the number of opposing lower-validity cues. Dual-Process assumes the probability of employing the non-compensatory Take-The-Best (TTB) strategy depends solely on the absolute validity of the top discriminating cue. Because the top cue is identical across the first four trials, the probability of executing TTB remains fixed. As the compensatory WADD fallback strategy increasingly favors the opposing option, the overall choice probability for the top-cue option will drop but then plateau at the exact probability of TTB selection (e.g., an intermediate asymptote like 60%). In contrast, Generalized WADD integrates all features; adding more opposing cues continuously increases the evidence sum for the alternative option, driving the probability of choosing the top-cue option progressively toward zero. A single gamma scaling parameter in WADD cannot produce a stable intermediate plateau across these escalating conflicts.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Dual-Process Strategy Selection Theory: Decision-makers probabilistically select between a non-compensatory heuristic (Take-The-Best) and a compensatory strategy (Weighted Additive, WADD) on a trial-by-trial basis. The probability of employing the non-compensatory heuristic is a logistic function of the absolute validity of the highest-ranking discriminating cue. When the top discriminating cue is highly valid, subjects are highly likely to rely solely on it (TTB). However, when the top discriminating cue is weak, confidence in the heuristic drops, and subjects fall back to integrating all available information (WADD).

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- theta: [0.0, 1.0]
- tau: [1.0, 100.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    theta = float(parameters["theta"])
    tau = float(parameters["tau"])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    top_cue = None
    for j in cue_order:
        if a[j] != b[j]:
            top_cue = j
            break
            
    if top_cue is None:
        p_mix = np.array([0.5, 0.5])
    else:
        v_top = val[top_cue]
        
        # Probability of using TTB is a logistic function of the top cue's validity
        p_ttb_use = 1.0 / (1.0 + np.exp(-tau * (v_top - theta)))
        
        # Take-The-Best (TTB) prediction
        winner_ttb = 0 if a[top_cue] > b[top_cue] else 1
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
        # Weighted Additive (WADD) prediction
        score_a = np.sum(val * a)
        score_b = np.sum(val * b)
        scores_wadd = np.array([score_a, score_b])
        z_wadd = beta * (scores_wadd - np.max(scores_wadd))
        e_wadd = np.exp(z_wadd)
        p_wadd = e_wadd / np.sum(e_wadd)
        
        # Mix the two strategies based on the top cue validity
        p_mix = p_ttb_use * p_ttb + (1.0 - p_ttb_use) * p_wadd
        
    # Apply lapse rate
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


## COMPETING THEORY
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
[0] rationale: This metric calculates the difference in preference for the single highly-valid cue option between Trial 1 and Trial 2. Dual-Process Theory predicts a large positive difference because a 0.90 cue triggers TTB (choosing the single cue in Trial 1), while a 0.80 cue triggers WADD fallback (choosing the coalition in Trial 2). Generalized WADD, however, requires a single gamma parameter; if gamma is high enough to favor the single 0.90 cue over a coalition in Trial 1, it mathematically MUST also favor the single 0.80 cue over a weaker coalition in Trial 2, resulting in a difference near zero or even negative.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    
    # Trial 1: A=[1, 0, 0, 0, 0], B=[0, 1, 1, 1, 0]
    t1_A = "10000"
    t1_B = "01110"
    
    # Trial 2: A=[0, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t2_A = "01000"
    t2_B = "00111"
    
    # Calculate probability of choosing the single-cue option in Trial 1
    t1_mask_1 = (data['A_str'] == t1_A) & (data['B_str'] == t1_B)
    t1_mask_2 = (data['A_str'] == t1_B) & (data['B_str'] == t1_A)
    p_A_t1 = 0.5
    if t1_mask_1.sum() > 0 or t1_mask_2.sum() > 0:
        chose_t1_A_1 = (data.loc[t1_mask_1, 'response'] == 0).sum()
        chose_t1_A_2 = (data.loc[t1_mask_2, 'response'] == 1).sum()
        p_A_t1 = (chose_t1_A_1 + chose_t1_A_2) / (t1_mask_1.sum() + t1_mask_2.sum())
        
    # Calculate probability of choosing the single-cue option in Trial 2
    t2_mask_1 = (data['A_str'] == t2_A) & (data['B_str'] == t2_B)
    t2_mask_2 = (data['A_str'] == t2_B) & (data['B_str'] == t2_A)
    p_A_t2 = 0.5
    if t2_mask_1.sum() > 0 or t2_mask_2.sum() > 0:
        chose_t2_A_1 = (data.loc[t2_mask_1, 'response'] == 0).sum()
        chose_t2_A_2 = (data.loc[t2_mask_2, 'response'] == 1).sum()
        p_A_t2 = (chose_t2_A_1 + chose_t2_A_2) / (t2_mask_1.sum() + t2_mask_2.sum())
        
    return float(p_A_t1 - p_A_t2)
outcome: self_sim=0.0705 (var=0.0263) adversary_sim=0.1284 (var=0.0276) welch_t=-1.246 p=0.2187 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates a difference-in-differences to measure how sensitive choice probabilities are to adding a single 0.60 validity cue to the coalition, depending on whether the single top cue has a validity of 0.90 or 0.80. The Dual-Process theory posits that a 0.90 cue strongly triggers non-compensatory TTB, making the subject insensitive to the coalition's size (thus diff_90 is near 0). A 0.80 cue, however, often triggers WADD fallback, making the subject highly sensitive to the coalition's size (thus diff_80 is large). Generalized WADD scales all validities globally via gamma, so the score difference is identical in both cases, leading to a much smaller double-difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    
    def get_pA(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        chose_A = 0
        total = 0
        if mask1.sum() > 0:
            chose_A += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_A += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
        return chose_A / total if total > 0 else 0.5

    # Top cue 0.9
    pA_t1 = get_pA("10000", "01110")
    pA_t3 = get_pA("10000", "01111")
    
    # Top cue 0.8
    pA_t2 = get_pA("01000", "00111")
    pA_t4 = get_pA("01000", "00110")
    
    # Difference in choice probability when adding one 0.60 cue to the coalition
    diff_80 = pA_t4 - pA_t2
    diff_90 = pA_t1 - pA_t3
    
    return float(diff_80 - diff_90)
outcome: self_sim=0.0600 (var=0.0257) adversary_sim=0.0105 (var=0.0361) welch_t=+0.995 p=0.325 (N=25, alpha=0.01) -> reject

[2] rationale: This metric calculates the difference in the probability of choosing Option A (the option favored by the top cue) between trials where the top cue has a validity of 0.90 and trials where the top cue has a validity of 0.80. In the Dual-Process theory, a 0.90 cue is much more likely to trigger Take-The-Best (which strongly favors A) compared to a 0.80 cue, leading to a positive difference. In Generalized WADD, for most plausible parameter values, the compensatory sums actually favor Option A slightly more (or equally) in the 0.80-top-cue trials because the opposing coalitions are weaker, leading to a negative or near-zero difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    
    def get_pA(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        chose_A = 0
        total = 0
        if mask1.sum() > 0:
            chose_A += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_A += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
        return chose_A / total if total > 0 else 0.5

    # Trials where top discriminating cue is 0.90
    pA_t1 = get_pA("10000", "01110")
    pA_t3 = get_pA("10000", "01111")
    pA_t5 = get_pA("11000", "00111")
    avg_pA_90 = (pA_t1 + pA_t3 + pA_t5) / 3.0
    
    # Trials where top discriminating cue is 0.80
    pA_t2 = get_pA("01000", "00111")
    pA_t4 = get_pA("01000", "00110")
    avg_pA_80 = (pA_t2 + pA_t4) / 2.0
    
    return float(avg_pA_90 - avg_pA_80)
outcome: self_sim=0.0784 (var=0.0289) adversary_sim=0.1221 (var=0.0179) welch_t=-1.010 p=0.3179 (N=25, alpha=0.01) -> reject

[3] rationale: This metric isolates the difference in preference for the single-cue option between Trial 3 and Trial 4. In Dual-Process theory, Trial 3's top cue is 0.90, which triggers a higher rate of Take-The-Best (favoring A) than Trial 4's top cue of 0.80. Thus, pA_t3 should be greater than pA_t4. In Generalized WADD, however, the compensatory sum actually favors Option A much earlier (at lower gammas) in Trial 4 (0.8 vs 1.2) than in Trial 3 (0.9 vs 2.6). Because WADD struggles to overcome the massive coalition in Trial 3 without a very high gamma, it will generally predict a lower probability of choosing A in Trial 3 compared to Trial 4, leading to a negative or near-zero difference, cleanly separating the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    
    def get_pA(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        chose_A = 0
        total = 0
        if mask1.sum() > 0:
            chose_A += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_A += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
        return chose_A / total if total > 0 else 0.5

    # Trial 3: Top cue 0.9 vs coalition of 0.8 + 3x0.6
    pA_t3 = get_pA("10000", "01111")
    
    # Trial 4: Top cue 0.8 vs coalition of 2x0.6
    pA_t4 = get_pA("01000", "00110")
    
    return float(pA_t3 - pA_t4)
outcome: self_sim=0.0568 (var=0.0341) adversary_sim=0.0674 (var=0.0495) welch_t=-0.182 p=0.8564 (N=25, alpha=0.01) -> reject

[4] rationale: This metric calculates the difference in preference for the single-cue option between Trial 1 (top cue = 0.96) and Trial 3 (top cue = 0.76). Dual-Process theory strongly predicts a positive difference because the higher absolute validity of 0.96 makes subjects much more likely to employ Take-The-Best (which strictly favors A) compared to the 0.76 cue. Conversely, Generalized WADD predicts a negative difference. Because validities are exponentiated by gamma, the ratio between consecutive validities increases as they get smaller (e.g., 0.66/0.76 > 0.86/0.96), making the top cue relatively stronger against its opposing coalition in Trial 3 than in Trial 1. This creates a perfect structural reversal: Dual-Process predicts P(A|T1) > P(A|T3), while WADD predicts P(A|T1) < P(A|T3).
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    
    def get_pA(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        chose_A = 0
        total = 0
        if mask1.sum() > 0:
            chose_A += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_A += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
        return chose_A / total if total > 0 else 0.5

    # Trial 1: Top cue 0.96 vs coalition of 0.86 + 0.76 + 0.66 + 0.56
    pA_t1 = get_pA("10000", "01111")
    
    # Trial 3: Top cue 0.76 vs coalition of 0.66 + 0.56
    pA_t3 = get_pA("00100", "00011")
    
    return float(pA_t1 - pA_t3)
outcome: self_sim=0.2050 (var=0.0930) adversary_sim=0.2017 (var=0.0621) welch_t=+0.042 p=0.9664 (N=25, alpha=0.01) -> reject

[5] rationale: This metric exploits a structural impossibility in the Generalized WADD model. In WADD, as the exponential scaling parameter (gamma) increases, the smaller opposing coalition in Trial 3 is overcome by its top cue BEFORE the larger opposing coalition in Trial 1 is overcome. Thus, WADD can never predict that Option A is favored in Trial 1 but opposed in Trial 3; it either favors B in both, A in both, or A in Trial 3 and B in Trial 1. Consequently, P(A|T1) - 2*P(A|T3) is robustly negative in WADD across its entire parameter space. Conversely, the Dual-Process theory posits that Trial 1's highly valid top cue (0.96) strongly triggers Take-The-Best (favoring A), while Trial 3's weaker top cue (0.76) triggers compensatory fallback (favoring B). This allows Dual-Process to easily predict P(A|T1) > 0.5 and P(A|T3) < 0.5, yielding a significantly more positive score on this metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    
    def get_pA(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        chose_A = 0
        total = 0
        if mask1.sum() > 0:
            chose_A += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_A += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
        return chose_A / total if total > 0 else 0.5

    # Trial 1: Top cue 0.96 vs coalition of 0.86 + 0.76 + 0.66 + 0.56
    pA_t1 = get_pA("10000", "01111")
    
    # Trial 3: Top cue 0.76 vs coalition of 0.66 + 0.56
    pA_t3 = get_pA("00100", "00011")
    
    # The metric contrasts the probability of choosing A in Trial 1 heavily against Trial 3.
    return float(pA_t1 - 2.0 * pA_t3)
outcome: self_sim=-0.4225 (var=0.3582) adversary_sim=-0.3600 (var=0.0932) welch_t=-0.465 p=0.6447 (N=25, alpha=0.01) -> reject

[6] rationale: This metric calculates the absolute curvature of the probability of choosing Option A across Trials 1, 2, and 3. The top cue validities in these trials decrease linearly (0.96, 0.86, 0.76). In the Generalized WADD model, the choice probabilities change relatively smoothly and linearly across these trials, resulting in a true curvature near zero (with observed curvature driven mostly by binomial noise). In the Dual-Process theory, the probability of using Take-The-Best is a logistic step-function of the top cue's validity. This S-curve introduces severe non-linearity: depending on the subject's threshold parameter (theta), the choice probability will drop sharply between either T1 and T2, or T2 and T3, while remaining flat elsewhere. Taking the absolute value captures this sharp structural non-linearity regardless of where the threshold lies, leading to a significantly higher expected value for Dual-Process compared to WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join(map(str, map(int, x))))
    
    def get_pA(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        chose_A = 0
        total = 0
        if mask1.sum() > 0:
            chose_A += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_A += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
        return chose_A / total if total > 0 else 0.5

    # Trial 1: Top cue 0.96
    pA_t1 = get_pA("10000", "01111")
    
    # Trial 2: Top cue 0.86
    pA_t2 = get_pA("01000", "00111")
    
    # Trial 3: Top cue 0.76
    pA_t3 = get_pA("00100", "00011")
    
    # Calculate the absolute second derivative (curvature) of P(A) across the three trials
    return float(abs(pA_t1 + pA_t3 - 2.0 * pA_t2))

outcome: self_sim=0.0333 (var=0.0665) adversary_sim=0.0400 (var=0.0256) welch_t=-0.110 p=0.9131 (N=25, alpha=0.01) -> reject

[7] rationale: Dual-Process theory predicts that a cue validity of 0.76 (Trial 3's top cue) is still strong enough to trigger the Take-The-Best heuristic for the majority of subjects (since the threshold theta is uniform in [0, 1]). Because TTB evaluates to a stark 1 vs 0 evidence signal regardless of absolute validity, DP predicts highly deterministic choices (far from 0.5) in both Trial 4 (top cue 0.96) and Trial 3. Generalized WADD, however, scales validities globally. Because the lower validities must sum to a tiny fraction of the total weight when gamma is high, the absolute difference in evidence for Trial 3 is mathematically forced to be near zero for almost all parameter values. Consequently, WADD predicts choice probabilities very close to 0.5 for Trial 3, while maintaining highly deterministic choices for Trial 4. By computing `abs(P(A|T4) - 0.5) - abs(P(A|T3) - 0.5)`, we measure this structural difference. DP evaluates to near zero (both trials are roughly equally deterministic), whereas WADD evaluates to a large positive value (Trial 4 is deterministic, Trial 3 is a coin toss).
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    def get_pA(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        chose_A = 0
        total = 0
        if mask1.sum() > 0:
            chose_A += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_A += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
        return chose_A / total if total > 0 else 0.5

    # Trial 4: Top cue is 0.96. Both models predict highly deterministic choices.
    pA_t4 = get_pA("10011", "01100")
    
    # Trial 3: Top cue is 0.76. 
    # DP predicts this is still strong enough to trigger deterministic TTB for most subjects.
    # WADD shrinks this difference to near-zero due to global normalization, predicting a coin toss.
    pA_t3 = get_pA("00100", "00011")
    
    # The metric contrasts the determinism (distance from 0.5) between the two trials.
    return float(abs(pA_t4 - 0.5) - abs(pA_t3 - 0.5))
outcome: self_sim=0.2492 (var=0.0204) adversary_sim=0.2833 (var=0.0210) welch_t=-0.840 p=0.4053 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    \n    def get_pA(a_str, b_str):\n        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)\n        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)\n        chose_A = 0\n        total = 0\n        if mask1.sum() > 0:\n            chose_A += (data.loc[mask1, 'response'] == 0).sum()\n            total += mask1.sum()\n        if mask2.sum() > 0:\n            chose_A += (data.loc[mask2, 'response'] == 1).sum()\n            total += mask2.sum()\n        return chose_A / total if total > 0 else 0.5\n\n    # Trial 2: Top cue vs coalition of 2 cues\n    pA_t2 = get_pA(\"10000\", \"01100\")\n    \n    # Trial 4: Top cue vs coalition of 4 cues\n    pA_t4 = get_pA(\"10000\", \"01111\")\n    \n    return float(pA_t2 - pA_t4)",
  "rationale": "This metric calculates the drop in probability of choosing the top-cue option between Trial 2 and Trial 4. In Dual-Process theory, the top cue is identical (0.90) across these trials, meaning the probability of employing the non-compensatory Take-The-Best strategy is fixed. Because the compensatory WADD fallback strategy already strongly favors the opposing coalition in Trial 2, adding more opposing cues in Trial 4 does not meaningfully change the overall choice probabilities. Thus, DP predicts a plateau where P(A|T2) - P(A|T4) is near zero. In contrast, Generalized WADD continuously integrates all features; adding two more opposing cues from Trial 2 to Trial 4 substantially increases the accumulated evidence for the alternative option, steadily driving P(A) further down. Consequently, WADD predicts a significantly positive difference, firmly separating the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 9812,
  "candidates_token_count": 613,
  "total_token_count": 16554
}
```
