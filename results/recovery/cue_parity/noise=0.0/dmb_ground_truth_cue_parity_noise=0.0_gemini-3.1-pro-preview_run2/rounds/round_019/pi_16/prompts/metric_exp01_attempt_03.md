# metric_exp01_attempt_03

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
**Validities (n_features=6):** [0.95, 0.85, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]

**Rationale:** This design quantitatively and qualitatively dissociates the advocated Leaky Competing Accumulator (LCA) theory from the competing Conflict-Modulated Threshold Sequential Sampling theory by exploiting a paradoxical non-monotonicity in the competing theory's threshold mechanism. The competing theory assumes that the decision threshold is modulated by 'conflict', which it defines as the inverse of the absolute tally difference between the options. Consequently, a large tally difference in any direction yields low conflict and a low threshold, triggering early stopping. By pitting a single extremely high-validity cue for Option A against a systematically increasing number of lower-validity cues for Option B, we increase the tally difference in favor of B. Paradoxically, the competing theory predicts that as B gains MORE supportive cues, the threshold drops, causing the model to stop searching immediately after sampling A's top cue and completely ignore B's evidence—resulting in a stronger preference for A. The advocated LCA model, which evaluates all cues simultaneously and configurally scales them, correctly predicts that adding supportive cues to B will monotonically increase B's accumulated evidence and decrease the preference for A.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Leaky Competing Accumulator with Non-linear Configural Weighting and Evidence Transduction: Decision-makers evaluate cues simultaneously, with evidence for each option dynamically inhibiting the other in a leaky competing accumulator (LCA). Cues have a configural impact, scaled non-linearly by the total number of supporting cues. Additionally, the integrated evidence for each option is passed through a non-linear transducer (alpha) before entering the accumulation process, allowing the model to amplify the differences driven by high-validity cues and capture strong non-compensatory reversals.

**Parameters:**
- gamma: [-10.0, 10.0]
- leak: [0.1, 2.0]
- inhibition: [0.0, 5.0]
- theta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- alpha: [0.1, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    leak = float(parameters["leak"])
    inhibition = float(parameters["inhibition"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    # Configural weighting: effective validity of a cue is non-linearly modulated by the total number of cues
    # Using max(1e-6, sum) to avoid 0^negative_gamma undefined errors
    sum_a_safe = max(1e-6, sum_a)
    sum_b_safe = max(1e-6, sum_b)
    
    w_a = val * (sum_a_safe ** gamma)
    w_b = val * (sum_b_safe ** gamma)
    
    # Make sure inputs are non-negative and apply non-linear transducer alpha
    I_A = max(0.0, np.sum(w_a * a)) ** alpha
    I_B = max(0.0, np.sum(w_b * b)) ** alpha
    
    # Leaky Competing Accumulator (LCA) simulation
    x_a, x_b = 0.0, 0.0
    dt = 0.1
    steps = 100
    
    for _ in range(steps):
        dx_a = (I_A - leak * x_a - inhibition * x_b) * dt
        dx_b = (I_B - leak * x_b - inhibition * x_a) * dt
        
        x_a = max(0.0, x_a + dx_a)
        x_b = max(0.0, x_b + dx_b)
        
    # Softmax choice based on final activations
    z = theta * np.array([x_a, x_b])
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** Sequential Cue Sampling with Conflict-Modulated Threshold: Decision-makers sample cues sequentially in order of their validities. Instead of integrating all cues simultaneously, they accumulate evidence until it reaches a decision threshold. Crucially, the level of conflict between the options (inversely related to the absolute tally difference) dynamically modulates this threshold. Under low conflict, the threshold is low, leading to fast, non-compensatory decisions (like Take-The-Best). Under high conflict, the threshold increases, enforcing extensive sampling and yielding compensatory behavior. This provides a natural speed-accuracy tradeoff without mathematical artifacts.

**Parameters:**
- theta_0: [0.0, 3.0]
- theta_1: [0.0, 5.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta_0 = float(parameters["theta_0"])
    theta_1 = float(parameters["theta_1"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate conflict based on the absolute tally difference of discriminating cues
    N_a = np.sum((a > b).astype(int))
    N_b = np.sum((b > a).astype(int))
    tally_diff = abs(N_a - N_b)
    
    # High tally difference means low conflict; zero tally difference means maximal conflict
    conflict = 1.0 / (1.0 + tally_diff)
    
    # Decision threshold is modulated by the level of conflict
    theta = theta_0 + theta_1 * conflict
    
    # Sequential sampling of cues prioritized by validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for j in cue_order:
        E += (a[j] - b[j]) * val[j]
        if abs(E) >= theta:
            break
            
    # Softmax choice probability based on accumulated evidence
    z = beta * np.array([E, -E])
    z = z - np.max(z)
    p = np.exp(z) / np.sum(np.exp(z))
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
[0] rationale: The experimental design groups trials into sets where the discriminating cues (cues that differ between option A and option B) remain exactly the same, but the number of tied cues (cues present for both options) increases. The competing Sequential Cue Sampling theory computes conflict and evidence based exclusively on discriminating cues, making it completely blind to tied cues; therefore, it predicts zero shift in choice probabilities across these trial pairs. In contrast, the advocated LCA theory with Configural Weighting scales the effective validity of all cues by the total sum of cues present for each option. Adding tied cues non-linearly alters these sums, causing a systematic shift in choice probabilities. This metric calculates the sum of absolute differences in choice proportions between the trials with the most tied cues and the trials with no tied cues within each set. For the competing theory, this metric will just capture binomial sampling noise (close to 0), whereas for the advocated theory, it will capture the systematic, graded shifts predicted by the configural weighting mechanism.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    def get_mean(a_str, b_str):
        mask = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        if mask.sum() == 0:
            return 0.5
        return data.loc[mask, 'response'].mean()

    # Trial 3 vs Trial 1
    diff1 = get_mean("10011", "01111") - get_mean("10000", "01100")
    # Trial 7 vs Trial 4
    diff2 = get_mean("10111", "01111") - get_mean("10000", "01000")
    # Trial 9 vs Trial 8
    diff3 = get_mean("10001", "01111") - get_mean("10000", "01110")
    
    return float(abs(diff1) + abs(diff2) + abs(diff3))
outcome: self_sim=0.2180 (var=0.1689) adversary_sim=0.0260 (var=0.0369) welch_t=+2.116 p=0.04174 (N=25, alpha=0.01) -> reject

[1] rationale: The experiment groups trials into sets where the discriminating cues remain constant but the number of tied cues increases. The competing Sequential Cue Sampling theory computes conflict and accumulates evidence based solely on discriminating cues, making it completely blind to tied cues; therefore, it predicts a flat line (slope = 0) for choice probabilities across these trial sets. In contrast, the advocated LCA theory with Configural Weighting scales the effective validity of all cues by the total sum of cues present. Adding tied cues non-linearly alters these sums, causing a systematic, graded shift in choice probabilities (non-zero slope). This metric computes the linear slope of choice proportions as a function of the number of tied cues for each trial group, combines them using an optimal inverse-variance weighted mean to minimize binomial noise, and takes the absolute value per subject (since the shift direction depends on the sign of the individual's gamma parameter). Finally, it averages these absolute slopes across subjects. For the competing theory, this yields a tightly bounded value near zero (driven only by minimal binomial noise), while for the advocated theory it robustly captures the strong, systematic shifts.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    def get_subj_metric(df):
        def get_mean(a_str, b_str):
            mask = (df['A_str'] == a_str) & (df['B_str'] == b_str)
            if mask.sum() == 0:
                return 0.5
            return float(df.loc[mask, 'response'].mean())
            
        # Group 1: trials 1, 2, 3 (0, 1, and 2 tied cues)
        g1_y0 = get_mean('10000', '01100')
        g1_y1 = get_mean('10010', '01110')
        g1_y2 = get_mean('10011', '01111')
        slope1 = (g1_y2 - g1_y0) / 2.0
        
        # Group 2: trials 4, 5, 6, 7 (0, 1, 2, and 3 tied cues)
        g2_y0 = get_mean('10000', '01000')
        g2_y1 = get_mean('10100', '01100')
        g2_y2 = get_mean('10110', '01110')
        g2_y3 = get_mean('10111', '01111')
        slope2 = (3.0 * g2_y3 + 1.0 * g2_y2 - 1.0 * g2_y1 - 3.0 * g2_y0) / 10.0
        
        # Group 3: trials 8, 9 (0 and 1 tied cues)
        g3_y0 = get_mean('10000', '01110')
        g3_y1 = get_mean('10001', '01111')
        slope3 = g3_y1 - g3_y0
        
        # Inverse-variance weighted mean of the slopes to minimize binomial noise
        # Variances are proportional to 0.5, 0.2, and 2.0 respectively.
        # Weights are 1/0.5=2.0, 1/0.2=5.0, 1/2.0=0.5. Sum of weights = 7.5
        weighted_mean_slope = (2.0 * slope1 + 5.0 * slope2 + 0.5 * slope3) / 7.5
        
        return abs(weighted_mean_slope)
        
    shifts = data.groupby('subject_id').apply(get_subj_metric)
    return float(shifts.mean())
outcome: self_sim=0.0531 (var=0.0034) adversary_sim=0.0373 (var=0.0010) welch_t=+1.180 p=0.2456 (N=25, alpha=0.01) -> reject

[2] rationale: The experimental design groups trials such that within each group, the discriminating cues are identical but the number of tied cues increases. The competing Sequential Cue Sampling theory computes evidence and conflict strictly from discriminating cues, predicting absolutely identical choice probabilities within each group (any observed variance is purely binomial noise). The advocated LCA theory with Configural Weighting predicts that tied cues non-linearly alter the effective weights of the discriminating cues, causing systematic shifts in choice probabilities within each group. By computing the standard deviation of the choice proportions within each group and summing them, this metric captures the presence of these systematic shifts. Using the standard deviation instead of raw absolute differences acts as a natural squashing function, significantly reducing between-subject variance while maintaining a strong contrast between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    def get_mean(a_str, b_str):
        mask = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        if mask.sum() == 0:
            return 0.5
        return data.loc[mask, 'response'].mean()

    # Group 1: 0, 1, and 2 tied cues
    g1 = [get_mean('10000', '01100'), 
          get_mean('10010', '01110'), 
          get_mean('10011', '01111')]
          
    # Group 2: 0, 1, 2, and 3 tied cues
    g2 = [get_mean('10000', '01000'), 
          get_mean('10100', '01100'), 
          get_mean('10110', '01110'), 
          get_mean('10111', '01111')]
          
    # Group 3: 0 and 1 tied cues
    g3 = [get_mean('10000', '01110'), 
          get_mean('10001', '01111')]
          
    # Calculate the standard deviation of choice proportions within each group.
    # Using standard deviation squashes extreme shifts, reducing between-subject variance
    # while still robustly capturing the non-zero variance predicted by the advocated theory.
    return float(np.std(g1) + np.std(g2) + np.std(g3))
outcome: self_sim=0.1083 (var=0.0365) adversary_sim=0.0275 (var=0.0103) welch_t=+1.868 p=0.06984 (N=25, alpha=0.01) -> reject

[3] rationale: Previous attempts failed due to the high between-subject variance of the advocated theory's predictions, which is driven by the uniform sampling of the configural weighting parameter (gamma). When gamma is large, the absolute shifts in choice probabilities are huge; when gamma is near zero, the shifts are tiny. To solve this, this metric computes the variance of choice proportions *within* the trial sets (where discriminating cues are constant) and divides it by the *total* variance across all trials. For the competing theory, the within-set variance is purely binomial noise, so this ratio remains low. For the advocated theory, the tied cues induce systematic within-set variance. By taking the ratio of within-set variance to total variance, we normalize out the absolute magnitude of the shifts, drastically stabilizing the metric across subjects while preserving a strong contrast between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    def get_mean(a_str, b_str):
        mask = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        if mask.sum() == 0:
            return 0.5
        return data.loc[mask, 'response'].mean()

    # Set 1: 0, 1, and 2 tied cues
    t1 = get_mean('10000', '01100')
    t2 = get_mean('10010', '01110')
    t3 = get_mean('10011', '01111')
    
    # Set 2: 0, 1, 2, and 3 tied cues
    t4 = get_mean('10000', '01000')
    t5 = get_mean('10100', '01100')
    t6 = get_mean('10110', '01110')
    t7 = get_mean('10111', '01111')
    
    # Set 3: 0 and 1 tied cues
    t8 = get_mean('10000', '01110')
    t9 = get_mean('10001', '01111')
    
    # Calculate within-set variances (shifts driven by tied cues)
    v1 = np.var([t1, t2, t3])
    v2 = np.var([t4, t5, t6, t7])
    v3 = np.var([t8, t9])
    v_within = v1 + v2 + v3
    
    # Calculate total variance across all trials
    v_total = np.var([t1, t2, t3, t4, t5, t6, t7, t8, t9])
    
    # Return the ratio of within-set variance to total variance.
    # This normalizes out the absolute magnitude of the shifts (which varies wildly due to gamma),
    # drastically reducing between-subject variance while isolating the structural difference.
    return float(v_within / (v_total + 0.01))
outcome: self_sim=0.0488 (var=0.8025) adversary_sim=0.0053 (var=0.1398) welch_t=+0.224 p=0.8243 (N=25, alpha=0.01) -> reject

[4] rationale: This metric calculates the difference in the probability of choosing Option B between trials where B has many cues (4 or 5 cues) and trials where B has few cues (1 or 2 cues). The advocated LCA theory predicts that adding more supportive cues to B increases its integrated evidence, leading to a higher probability of choosing B (a strongly positive difference). In contrast, the competing Sequential Cue Sampling theory predicts a paradox: adding more supportive cues to B increases the absolute tally difference, which reduces conflict and drops the decision threshold. This lower threshold causes the model to stop searching immediately after evaluating Option A's highly valid first cue, completely ignoring B's additional cues. Thus, the competing theory predicts a lower or equal probability of choosing B when B has more cues (a negative or near-zero difference). This fundamental divergence in the direction of the effect yields a robust and highly discriminative metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['B_sum'] = data['option_b_ratings'].apply(sum)
    
    subj_high = data[data['B_sum'] >= 4].groupby('subject_id')['response'].mean()
    subj_low = data[data['B_sum'] <= 2].groupby('subject_id')['response'].mean()
    
    return float((subj_high - subj_low).mean())
outcome: self_sim=0.2090 (var=0.0411) adversary_sim=0.2523 (var=0.0328) welch_t=-0.797 p=0.4293 (N=25, alpha=0.01) -> reject

[5] rationale: The simplest and most robust point of divergence between the two theories on this experimental design is the overall rate of choosing Option B. The advocated LCA theory evaluates all cues simultaneously; because Option B consistently has a higher total sum of validities across most trials, LCA frequently accumulates more evidence for B (especially when gamma >= 0), leading to a relatively high overall probability of choosing B. In contrast, the competing Sequential Cue Sampling theory strictly prioritizes the most valid cue (which always belongs to Option A). Because the increasing number of B's cues paradoxically drops the decision threshold by reducing conflict, the competing model is highly prone to stopping immediately after evaluating A's top cue, thereby ignoring B's abundant evidence entirely. This structural difference results in a substantially lower overall rate of choosing Option B for the competing theory. Using the grand mean choice proportion minimizes within-subject noise and provides a highly stable, tight metric for the Welch's t-test.
metric_source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
outcome: self_sim=0.5140 (var=0.0935) adversary_sim=0.7008 (var=0.0331) welch_t=-2.626 p=0.01227 (N=25, alpha=0.01) -> reject

[6] rationale: The overall rate of choosing Option B is a fundamental divergence point, but the advocated LCA theory exhibits high between-subject variance. This is because its configural weighting parameter (gamma) is symmetrically distributed around zero: when gamma > 0, LCA strongly prefers B (~0.9); when gamma < 0, it strongly prefers A (~0.1). By calculating the absolute deviation of each subject's mean choice proportion from 0.5, we 'fold' this bimodal distribution in half. This elegantly neutralizes the nuisance variance introduced by gamma, yielding a highly stable metric concentrated around 0.375 for LCA. In contrast, the competing Sequential Cue Sampling theory predominantly prefers Option B across most trials (averaging ~0.75), yielding an absolute deviation concentrated around 0.25. This metric preserves the structural gap between the theories while drastically crushing between-subject variance, ensuring robust statistical discrimination at the human sample size.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the mean response (proportion of choosing Option B) for each subject
    subj_means = data.groupby('subject_id')['response'].mean()
    
    # Calculate the absolute deviation from 0.5 for each subject
    # This 'folds' the distribution, neutralizing variance from symmetric parameters
    subj_dev = subj_means.apply(lambda x: abs(x - 0.5))
    
    # Return the average absolute deviation across all subjects
    return float(subj_dev.mean())
outcome: self_sim=0.2935 (var=0.0075) adversary_sim=0.2608 (var=0.0054) welch_t=+1.437 p=0.1573 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # Calculate the mean response for each of the 6 trial types per subject\n    trial_means = data.groupby(['subject_id', 'B_str'])['response'].mean()\n    \n    # Calculate the standard deviation of these 6 means for each subject\n    subj_std = trial_means.groupby('subject_id').std()\n    \n    return float(subj_std.mean())",
  "rationale": "By calculating the standard deviation of the mean choice proportions across the 6 distinct trial types for each subject, we capture the within-subject sensitivity to the changing cues of Option B. The advocated LCA theory predicts a monotonic, graded shift in choice probabilities as the number of Option B cues increases, leading to a robust standard deviation across trial types. In contrast, the competing Sequential Cue Sampling theory's threshold mechanism often leads to all-or-nothing step functions (either evaluating all cues or stopping immediately at cue 1), producing a different variance profile. Taking the standard deviation per subject naturally normalizes out the absolute direction of the preference (which varies wildly in LCA due to the symmetric gamma parameter) and provides a highly stable scalar summary of the structural differences."
}
```

## Usage

```json
{
  "prompt_token_count": 8118,
  "candidates_token_count": 321,
  "total_token_count": 13281
}
```
