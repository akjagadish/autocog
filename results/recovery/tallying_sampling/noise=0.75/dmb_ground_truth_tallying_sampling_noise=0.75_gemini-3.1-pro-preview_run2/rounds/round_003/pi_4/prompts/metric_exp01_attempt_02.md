# metric_exp01_attempt_02

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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** To quantitatively dissociate Probabilistic Search Take-The-Best (PS-TTB) from WADD-Gamma, we exploit WADD-Gamma's parameterization, which uses a power parameter (gamma <= 0.75) to compress the log-odds of validities. This compression makes WADD-Gamma highly compensatory, reliably favoring options with a larger number of lower-validity cues. In contrast, PS-TTB probabilistically searches through cues based on their validities and stops at the first discriminating cue, maintaining a strong non-compensatory characteristic. We use a 5-feature design with a steep validity gradient. We include 'conflict' trials where the highest-validity cue points to one option, but the remaining cues point to the other (WADD-Gamma favors the latter, PS-TTB favors the former). We also include tied highest-cue trials to test secondary cue reliance and agreement trials to accurately estimate baseline noise and search parameters.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Search Take-The-Best (PS-TTB)

**Parameters:**
- tau: [0.01, 100.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    diff = stim[0] - stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    tau = float(parameters['tau'])
    epsilon = float(parameters['epsilon'])
    
    n_features = len(validities)
    n_samples = 1000
    
    # Gumbel-max trick to sample permutations without replacement
    # probabilities proportional to softmax(validities / tau)
    logits = validities / (tau + 1e-6)
    gumbels = np.random.gumbel(size=(n_samples, n_features))
    orders = np.argsort(-(logits + gumbels), axis=1)
    
    diff_sign = np.sign(diff)
    ordered_diffs = diff_sign[orders]
    
    # Find the first discriminating cue in each sampled search order
    abs_diffs = np.abs(ordered_diffs)
    first_non_zero_idx = np.argmax(abs_diffs, axis=1)
    has_non_zero = np.any(abs_diffs > 0, axis=1)
    
    first_non_zero_vals = ordered_diffs[np.arange(n_samples), first_non_zero_idx]
    
    wins_a = np.sum((first_non_zero_vals == 1) & has_non_zero)
    wins_b = np.sum((first_non_zero_vals == -1) & has_non_zero)
    
    total = wins_a + wins_b
    if total > 0:
        p = np.array([wins_a / total, wins_b / total])
    else:
        p = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p + epsilon * (np.ones(2) / 2.0)
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
**Description:** Weighted Additive Model with Power-Scaled Log-Odds (WADD-Gamma). Decision-makers compute a weighted sum of features for each option. The weights are derived from the log-odds of the cue validities, raised to a power gamma. This parameterization allows the model to smoothly interpolate between Tallying/Equal-Weighting (gamma = 0) and standard log-odds WADD (gamma = 1). Choices are then made via a softmax over the weighted sums, incorporating an independent lapse rate for noise.

**Parameters:**
- gamma: [0.0, 0.75]
- beta: [0.0, 5.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Clip validities to ensure log-odds are strictly positive and well-defined
    v_clipped = np.clip(validities, 0.5001, 0.9999)
    log_odds = np.log(v_clipped / (1.0 - v_clipped))
    
    gamma = float(parameters["gamma"])
    weights = log_odds ** gamma
    
    # Weighted sum for each option
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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
[0] rationale: This metric calculates the proportion of times the subject chooses the option that possesses ONLY the highest-validity feature (feature 1), when pitted against an option that lacks feature 1 but possesses multiple lower-validity features (Trials 1, 6, and 7). PS-TTB, being non-compensatory and probabilistically searching based on validity, will heavily favor the single-cue option because it usually evaluates the most valid feature first. WADD-Gamma, due to its power-scaled log-odds, compresses weights such that multiple lower-validity cues easily sum to outweigh the single highest-validity cue, causing it to frequently choose the multi-cue option.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_f1 = np.array([x[0] for x in data['option_a_ratings']])
    a_sum = np.array([sum(x) for x in data['option_a_ratings']])
    
    b_f1 = np.array([x[0] for x in data['option_b_ratings']])
    b_sum = np.array([sum(x) for x in data['option_b_ratings']])
    
    # Identify trials where one option has ONLY the most valid feature (f1=1, sum=1)
    # and the other option lacks this feature (f1=0)
    a_is_target = (a_f1 == 1) & (a_sum == 1) & (b_f1 == 0)
    b_is_target = (b_f1 == 1) & (b_sum == 1) & (a_f1 == 0)
    
    target_mask = a_is_target | b_is_target
    if not np.any(target_mask):
        return 0.5
        
    responses = data['response'].values
    
    chose_target = np.zeros(len(data), dtype=bool)
    chose_target[a_is_target & (responses == 0)] = True
    chose_target[b_is_target & (responses == 1)] = True
    
    return float(np.mean(chose_target[target_mask]))

outcome: self_sim=0.3656 (var=0.0149) adversary_sim=0.3617 (var=0.0215) welch_t=+0.102 p=0.9192 (N=25, alpha=0.01) -> reject

[1] rationale: This metric isolates the difference in the probability of choosing Option A between specifically matched trial pairs (Trial 2 vs 3, and Trial 7 vs 6). In these pairs, the only difference is the presence of the highest-validity cue (Cue 1). For PS-TTB, adding or removing Cue 1 often leaves the ultimate choice unchanged because the model evaluates cues sequentially and will simply fall back to the next highly valid cue (which points in the same direction) or randomly select among remaining cues with a similar ratio. Thus, PS-TTB predicts a small difference in P(A) between these paired trials. Conversely, WADD-Gamma's compensatory nature means the addition of the highest-validity cue causes a massive swing in the weighted sum, shifting its preference from strongly favoring Option B to strongly favoring Option A (or vice versa). By summing these differences, the metric captures WADD-Gamma's large sensitivity to the compensatory sum while remaining near zero for PS-TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_f1 = np.array([x[0] for x in data['option_a_ratings']])
    a_f2 = np.array([x[1] for x in data['option_a_ratings']])
    a_f3 = np.array([x[2] for x in data['option_a_ratings']])
    b_f1 = np.array([x[0] for x in data['option_b_ratings']])
    b_f2 = np.array([x[1] for x in data['option_b_ratings']])
    
    # Trial 2: A=[1, 1, 0, 0, 0] B=[0, 0, 1, 1, 1]
    t2_mask = (a_f1 == 1) & (a_f2 == 1) & (b_f1 == 0) & (b_f2 == 0) & (a_f3 == 0)
    # Trial 3: A=[0, 1, 0, 0, 0] B=[0, 0, 1, 1, 1]
    t3_mask = (a_f1 == 0) & (a_f2 == 1) & (b_f1 == 0) & (b_f2 == 0) & (a_f3 == 0)
    
    # Trial 7: A=[0, 1, 0, 1, 1] B=[1, 0, 0, 0, 0]
    t7_mask = (a_f1 == 0) & (a_f2 == 1) & (b_f1 == 1) & (b_f2 == 0)
    # Trial 6: A=[0, 0, 1, 1, 1] B=[1, 0, 0, 0, 0]
    t6_mask = (a_f1 == 0) & (a_f2 == 0) & (b_f1 == 1) & (b_f2 == 0) & (a_f3 == 1)

    def p_a(mask):
        if np.sum(mask) == 0: return 0.5
        return np.mean(data['response'].values[mask] == 0)
        
    return float((p_a(t2_mask) - p_a(t3_mask)) + (p_a(t7_mask) - p_a(t6_mask)))
outcome: self_sim=0.0300 (var=0.0735) adversary_sim=0.2200 (var=0.1230) welch_t=-2.143 p=0.03755 (N=25, alpha=0.01) -> reject

[2] rationale: This metric isolates the effect of substituting Cue 2 for Cue 3 while holding both the 'best available cue' and the 'total number of cues' strictly constant for both options. In Trial 2 vs 5, and Trial 7 vs 6, the options are structurally identical except for the swap of the second and third most valid cues. For PS-TTB, whether the search is highly directed (small tau) or completely random (large tau), the choice probabilities remain virtually identical across these pairs because the best cue is unchanged and the cue counts are matched, driving the expected difference to near zero. In contrast, WADD-Gamma's compensatory weighted sum is highly sensitive to the exact log-odds weights of the cues; swapping Cue 2 for Cue 3 produces a consistent, substantial positive shift in the probability of choosing Option A. This yields a tight, near-zero distribution for PS-TTB and a distinct positive distribution for WADD-Gamma.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    # Trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
    t2_mask = (a_str == "11000") & (b_str == "00111")
    # Trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
    t5_mask = (a_str == "10100") & (b_str == "01011")
    
    # Trial 6: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
    t6_mask = (a_str == "00111") & (b_str == "10000")
    # Trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
    t7_mask = (a_str == "01011") & (b_str == "10000")
    
    def p_a(mask):
        if not np.any(mask):
            return 0.5
        return np.mean(data.loc[mask, 'response'] == 0)
        
    return float((p_a(t2_mask) - p_a(t5_mask)) + (p_a(t7_mask) - p_a(t6_mask)))
outcome: self_sim=-0.0567 (var=0.0935) adversary_sim=0.0767 (var=0.0819) welch_t=-1.592 p=0.118 (N=25, alpha=0.01) -> reject

[3] rationale: This metric leverages a double-difference of structurally paired trials to suppress within-model variance while maximizing the contrast between the non-compensatory probabilistic search of PS-TTB and the compensatory power-scaled summation of WADD-Gamma. 

For PS-TTB, the probability of choosing A in Trial 6 is almost identical to Trial 7, and Trial 2 is almost identical to Trial 4, regardless of the search noise parameter (tau). This is because swapping adjacent-validity cues (f2 for f3) barely alters the likelihood that the first randomly encountered discriminating cue favors the same option. Thus, PS-TTB expects this metric to be extremely close to 0.0 (ranging tightly between -0.1 and 0.0 across all tau values).

In contrast, WADD-Gamma evaluates options by summing power-scaled log-odds. Because the weights are non-linear, swapping cue 2 for cue 3 produces massive shifts in the weighted sums that frequently flip the model's categorical preference. Depending on the exact power parameter (gamma), WADD-Gamma will strongly favor Option B in one trial and Option A in its paired counterpart, driving the double-difference strongly negative (ranging between -1.0 and -0.5). This provides a large mean gap with minimal overlap.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    # T6: A=[0,0,1,1,1] B=[1,0,0,0,0]
    t6_mask = (a_str == "00111") & (b_str == "10000")
    # T7: A=[0,1,0,1,1] B=[1,0,0,0,0]
    t7_mask = (a_str == "01011") & (b_str == "10000")
    
    # T2: A=[1,1,0,0,0] B=[0,0,1,1,1]
    t2_mask = (a_str == "11000") & (b_str == "00111")
    # T4: A=[0,1,1,0,0] B=[0,0,0,1,1]
    t4_mask = (a_str == "01100") & (b_str == "00011")
    
    def p_a(mask):
        if not np.any(mask):
            return 0.5
        return np.mean(data.loc[mask, 'response'] == 0)
        
    # Metric: (P(A|T6) - P(A|T7)) + (P(A|T2) - P(A|T4))
    return float((p_a(t6_mask) - p_a(t7_mask)) + (p_a(t2_mask) - p_a(t4_mask)))
outcome: self_sim=0.0300 (var=0.0833) adversary_sim=-0.1417 (var=0.0912) welch_t=+2.055 p=0.04535 (N=25, alpha=0.01) -> reject

[4] rationale: This metric captures the fundamental difference between a non-compensatory model (PS-TTB) and a compensatory model (WADD-Gamma) by measuring how the choice probability changes when lower-validity cues are shifted to support the option with the highest available cue. In Trials 1 and 3, the target option has the single best available cue, but the competing option has a larger number of lower-validity cues. PS-TTB immediately selects the target option because it searches sequentially and stops at the best cue. Thus, when we add more supporting cues to the target option in Trials 2 and 8, PS-TTB's preference is already at the ceiling and shows little to no increase (difference ~ 0). Conversely, WADD-Gamma's compensatory nature means the multiple lower-validity cues in Trials 1 and 3 outweigh the single best cue, leading it to favor the competitor. However, when the lower-validity cues are shifted to the target option in Trials 2 and 8, the weighted sum flips the decision boundary, causing a massive increase in the probability of choosing the target option. This yields a tight, near-zero distribution for PS-TTB and a large positive value for WADD-Gamma.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    def p_target(target_a, target_b):
        m_straight = (a_str == target_a) & (b_str == target_b)
        m_swap = (a_str == target_b) & (b_str == target_a)
        
        n_straight = m_straight.sum()
        n_swap = m_swap.sum()
        
        if n_straight + n_swap == 0:
            return 0.5
            
        wins = 0
        if n_straight > 0:
            wins += (data.loc[m_straight, 'response'] == 0).sum()
        if n_swap > 0:
            wins += (data.loc[m_swap, 'response'] == 1).sum()
            
        return float(wins / (n_straight + n_swap))
        
    # T1 -> T2: Target option gets stronger (adds f3), competitor gets weaker
    p_t1 = p_target("10000", "01110")
    p_t2 = p_target("10100", "01011")
    
    # T3 -> T8: Target option gets stronger (adds f4), competitor gets weaker
    p_t3 = p_target("01000", "00111")
    p_t8 = p_target("01010", "00101")
    
    return float((p_t2 - p_t1) + (p_t8 - p_t3))
outcome: self_sim=0.2000 (var=0.1081) adversary_sim=0.4500 (var=0.1603) welch_t=-2.413 p=0.01984 (N=25, alpha=0.01) -> reject

[5] rationale: This metric exploits a mathematical invariance in PS-TTB that is systematically violated by WADD-Gamma. PS-TTB samples cues proportionally to exp(validity / tau). Because the validities [0.95, 0.85, 0.75, 0.65, 0.55] are evenly spaced, the log-odds used for sampling have a constant difference of 0.1/tau between adjacent cues. Trial 3 and Trial 4 present the exact same cue differences ([0, 1, -1, -1, -1]), which is a perfect 1-index shift of Trial 1 ([1, -1, -1, -1, 0]). Due to the constant difference in sampling logits, the probability of sampling the single positive cue before any negative cues is exactly identical between these trials for ANY search noise (tau) or lapse rate (epsilon). Thus, PS-TTB predicts `(P(A|T3) + P(A|T4))/2 - P(A|T1) = 0` with zero variance other than binomial noise. In contrast, WADD-Gamma uses `log_odds^gamma` as weights. Because this transformation is concave, the weight differences decrease for lower validities. The shift from T1 to T3/T4 increases the WADD-Gamma score by `-w1 + 2*w2 - w5`, which is strictly positive for all gamma > 0. Thus, WADD-Gamma predicts a strictly positive difference, perfectly separating the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    # Trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
    t1_mask = (a_str == "10000") & (b_str == "01110")
    
    # Trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
    t3_mask = (a_str == "01000") & (b_str == "00111")
    
    # Trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
    t4_mask = (a_str == "11000") & (b_str == "10111")
    
    def p_a(mask):
        if not np.any(mask):
            return 0.5
        return np.mean(data.loc[mask, 'response'] == 0)
        
    p_t1 = p_a(t1_mask)
    p_t3 = p_a(t3_mask)
    p_t4 = p_a(t4_mask)
    
    return float((p_t3 + p_t4) / 2.0 - p_t1)
outcome: self_sim=-0.0025 (var=0.0258) adversary_sim=0.0492 (var=0.0262) welch_t=-1.133 p=0.2627 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    a_str = data['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    b_str = data['option_b_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    \n    def p_target(target_a, target_b):\n        m_straight = (a_str == target_a) & (b_str == target_b)\n        m_swap = (a_str == target_b) & (b_str == target_a)\n        \n        n_straight = m_straight.sum()\n        n_swap = m_swap.sum()\n        \n        if n_straight + n_swap == 0:\n            return 0.5\n            \n        wins = 0\n        if n_straight > 0:\n            wins += (data.loc[m_straight, 'response'] == 0).sum()\n        if n_swap > 0:\n            wins += (data.loc[m_swap, 'response'] == 1).sum()\n            \n        return float(wins / (n_straight + n_swap))\n        \n    # Trial 1: Target A = 10000, Target B = 01110\n    p_t1 = p_target(\"10000\", \"01110\")\n    \n    # Trial 7: Target A = 10110, Target B = 01001\n    p_t7 = p_target(\"10110\", \"01001\")\n    \n    return float(p_t7 - p_t1)",
  "rationale": "This metric measures the change in preference for the option possessing the highest-validity cue (Cue 1) when it is supported by additional lower-validity cues, compared to when it stands alone against a strong compensatory alternative. In Trial 1, Option A has only Cue 1, while Option B has Cues 2, 3, and 4. WADD-Gamma, being compensatory, reliably favors Option B because the sum of the lower validities outweighs Cue 1. PS-TTB, being non-compensatory, searches sequentially by validity and frequently stops at Cue 1, thus favoring Option A. In Trial 7, Option A has Cues 1, 3, and 4, while Option B has Cues 2 and 5. Here, WADD-Gamma's weighted sum shifts massively in favor of Option A. PS-TTB also favors Option A, as it still possesses the highest validity cue. By taking the difference in the probability of choosing Option A between Trial 7 and Trial 1, we isolate the models' structural differences. For PS-TTB, the probability of choosing Option A is high in both trials (especially for small/moderate tau), leading to a tight difference near 0 (ranging up to 0.35 for fully random large tau). For WADD-Gamma, the probability swings from strongly favoring B in Trial 1 to strongly favoring A in Trial 7, yielding a large positive difference (typically 0.6 to 1.0). This provides a large, consistent mean gap with minimal overlap and low between-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 8025,
  "candidates_token_count": 784,
  "total_token_count": 40262
}
```
