# metric_exp02_attempt_02

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
**Validities (n_features=6):** [0.9, 0.85, 0.7, 0.7, 0.6, 0.6]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 6: A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0]

**Rationale:** This design quantitatively dissociates the advocated Sequential Evidence Accumulation theory from the competing Leaky Competing Accumulator (LCA) theory by exploiting their divergent handling of balanced, distinct opposing evidence. The advocated theory calculates evidence as a running difference between options. If two lower-validity cues have identical validities, assigning one to Option A and the other to Option B produces perfectly canceling evidence, leaving the net accumulated evidence unchanged. Consequently, the advocated theory predicts identical choice probabilities across trials where the core discriminating cues are fixed, regardless of how many balanced pairs of distinct lower-validity cues are added (or whether they trigger early stopping). The competing LCA theory, however, accumulates absolute evidence in parallel. Adding distinct but balanced cues increases the total input to both options' accumulators. Due to the LCA's mutual inhibition and non-linear zero-bounding, this higher total input alters the steady-state activation difference between the options, predicting systematic shifts in choice probability across these trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Evidence Accumulation: Decision-makers evaluate cues sequentially in descending order of validity. Each cue provides evidence proportional to a non-linear transformation of its validity above chance. Evidence is accumulated as a running difference between the two options. If the absolute accumulated evidence exceeds a threshold, search is terminated and a choice is made based on the current evidence. If all cues are evaluated without crossing the threshold, a decision is made based on the final accumulated evidence. This allows for fast, non-compensatory decisions when top cues are highly valid, while gracefully falling back to compensatory integration when early cues are less decisive.

**Parameters:**
- theta: [0.0, 10.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale weights by transforming validity above chance, allowing better separation
    weights = np.maximum(val - 0.5, 0.001) ** gamma
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if absolute accumulated evidence reaches the threshold
            if abs(E) >= theta:
                break
            
    scores = np.array([E, -E])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Decision-makers evaluate all cues in parallel using a Leaky Competing Accumulator (LCA) model. Evidence from each cue feeds into an option's accumulator proportional to a non-linear function of its validity above chance. The accumulators for the two options mutually inhibit each other and leak over time. This continuous, parallel competition naturally produces graded compensatory behavior: chance-level cues provide zero net evidence, while multiple weak opposing cues can slowly overcome a single strong cue's advantage through lateral inhibition.

**Parameters:**
- gamma: [0.1, 15.0]
- lambda_leak: [0.0, 5.0]
- beta_inhibit: [0.0, 5.0]
- tau: [0.1, 50.0]
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
    leak = float(parameters["lambda_leak"])
    inhibit = float(parameters["beta_inhibit"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear transformation of validity above chance
    weights = np.maximum(val - 0.5, 0.0) ** gamma
    
    # Total input evidence for each option
    I_A = np.sum(weights * a)
    I_B = np.sum(weights * b)
    
    # Simulate the Leaky Competing Accumulator
    x_A, x_B = 0.0, 0.0
    dt = 0.1
    for _ in range(50):
        dx_A = I_A - leak * x_A - inhibit * x_B
        dx_B = I_B - leak * x_B - inhibit * x_A
        
        x_A = max(0.0, x_A + dt * dx_A)
        x_B = max(0.0, x_B + dt * dx_B)
        
    scores = np.array([x_A, x_B])
    
    # Softmax choice rule
    z = tau * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
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
[0] rationale: This metric calculates the sensitivity of the decision-maker to the opposing evidence from lower-validity cues. By fixing the top cue favoring Option A and comparing the choice probability when Option B has no cues versus when Option B has all remaining cues, we can distinguish the two models. The Sequential Evidence Accumulation model predicts early stopping when the top cue is processed, leading to a flat choice probability curve (difference near 0). The Leaky Competing Accumulator evaluates all cues in parallel, so the addition of opposing cues for Option B will systematically shift choices toward B, leading to a large positive difference in response rates.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_sum'] = data['option_b_ratings'].apply(sum)
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        subj_diff = 0
        valid_a = 0
        for a_str in ['10000', '01000', '00100']:
            df_a = subj_df[subj_df['A_str'] == a_str]
            if df_a.empty: continue
            
            min_b = df_a['B_sum'].min()
            max_b = df_a['B_sum'].max()
            
            if min_b == max_b: continue
            
            resp_min = df_a[df_a['B_sum'] == min_b]['response'].mean()
            resp_max = df_a[df_a['B_sum'] == max_b]['response'].mean()
            
            subj_diff += (resp_max - resp_min)
            valid_a += 1
            
        if valid_a > 0:
            diffs.append(subj_diff / valid_a)
            
    if not diffs:
        return 0.0
    return float(np.mean(diffs))
outcome: self_sim=0.0700 (var=0.0422) adversary_sim=0.0200 (var=0.0432) welch_t=+0.856 p=0.3965 (N=25, alpha=0.01) -> reject

[1] rationale: Instead of looking at the difference between minimum and maximum opposing evidence, this metric directly measures the absolute propensity to choose the compensatory option (Option B) when it possesses all available lower-validity cues against a single top cue (Option A). The Sequential Evidence Accumulation model, depending on the threshold and gamma, can completely flip its decision to Option B if the threshold is high enough to allow the sum of weaker cues to overcome the top cue. In contrast, the LCA model's lateral inhibition tends to suppress the lower-validity cues' accumulators once the top cue gives Option A an early lead, keeping the response rate for Option B consistently lower and more stable.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask = (
        ((data['A_str'] == '10000') & (data['B_str'] == '01111')) |
        ((data['A_str'] == '01000') & (data['B_str'] == '00111')) |
        ((data['A_str'] == '00100') & (data['B_str'] == '00011'))
    )
    
    if mask.sum() == 0:
        return 0.0
        
    subj_means = data[mask].groupby('subject_id')['response'].mean()
    return float(subj_means.mean())

outcome: self_sim=0.4842 (var=0.0235) adversary_sim=0.4725 (var=0.0210) welch_t=+0.277 p=0.7833 (N=25, alpha=0.01) -> reject

[2] rationale: This metric exploits a fundamental dynamical difference between the models: lateral suppression. In the Leaky Competing Accumulator (LCA), lateral inhibition acts as a winner-take-all mechanism. When Option B's input is strictly weaker than Option A's (e.g., Trial 2, 7, and 11), B's accumulator is often completely suppressed to zero by A's stronger input. As a result, LCA predicts that adding a single weak opposing cue will have almost zero effect on the choice probability, making the drop between the baseline (no opposing cues) and the single-weak-cue trial near zero. In contrast, the Sequential Evidence Accumulation (SEA) model evaluates the evidence linearly (since its threshold is rarely reached early given the parameter priors). In SEA, adding a weak opposing cue strictly decreases the net accumulated evidence, leading to a consistent, positive drop in the probability of choosing A. By summing these specific drops, we isolate LCA's suppression mechanism from SEA's linear integration.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    data = data.copy()
    
    # Project list columns to string for hashability
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    subj_drops = []
    for subj, df in data.groupby('subject_id'):
        def get_p_A(a, b):
            mask = (df['A_str'] == a) & (df['B_str'] == b)
            if not mask.any():
                return 0.5
            # response is 0 for A, 1 for B
            return 1.0 - df[mask]['response'].mean()
            
        # Drop in P(A) when a single strictly weaker opposing cue is added
        drop1 = get_p_A('10000', '00000') - get_p_A('10000', '01000')
        drop2 = get_p_A('01000', '00000') - get_p_A('01000', '00100')
        drop3 = get_p_A('00100', '00000') - get_p_A('00100', '00010')
        
        subj_drops.append(drop1 + drop2 + drop3)
        
    return float(pd.Series(subj_drops).mean())
outcome: self_sim=0.0600 (var=0.1770) adversary_sim=-0.0100 (var=0.2049) welch_t=+0.566 p=0.5738 (N=25, alpha=0.01) -> reject

[3] rationale: This metric tests the fundamental difference between the linear evidence accumulation of SEA and the non-linear lateral inhibition of LCA. By converting choice probabilities to smoothed log-odds, we linearize the softmax function, directly exposing the underlying evidence dynamics. In SEA, the evidence is a linear sum of cue weights (unless it stops early, in which case drops are 0). Therefore, the drop in log-odds when a highly valid cue is added to Option B (e.g., Trial 1 to 2) will mathematically be strictly larger than the drop when a weakly valid cue is added later (e.g., Trial 4 to 5). This makes the metric (Drop_Early - Drop_Late) strictly positive for SEA. In contrast, LCA employs lateral inhibition, creating a winner-take-all dynamic. When Option B's evidence is weak (early trials), it is completely suppressed by Option A's strong top cue, resulting in near-zero drops in choice probability. However, as Option B accumulates multiple cues (later trials), it overcomes the inhibition, causing a sudden, massive drop in Option A's probability. This makes the late drop larger than the early drop, resulting in a negative or near-zero metric for LCA. Summing this across all three baseline sequences provides a robust, low-variance discriminator.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    # Project list columns to string for hashability
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    subj_metrics = []
    for subj, df in data.groupby('subject_id'):
        def get_L(a, b):
            mask = (df['A_str'] == a) & (df['B_str'] == b)
            if not mask.any():
                count_A = 4
                n = 8
            else:
                # response == 0 means Option A was chosen
                count_A = (df[mask]['response'] == 0).sum()
                n = len(df[mask])
            # Laplace smoothing to prevent log(0) and stabilize variance
            p = (count_A + 1) / (n + 2)
            return np.log(p / (1 - p))
            
        # Sequence 1: Top cue is w1
        L1 = get_L('10000', '00000')
        L2 = get_L('10000', '01000') # Drop effect of w2
        L4 = get_L('10000', '01110')
        L5 = get_L('10000', '01111') # Drop effect of w5
        m1 = (L1 - L2) - (L4 - L5)
        
        # Sequence 2: Top cue is w2
        L6 = get_L('01000', '00000')
        L7 = get_L('01000', '00100') # Drop effect of w3
        L8 = get_L('01000', '00110')
        L9 = get_L('01000', '00111') # Drop effect of w5
        m2 = (L6 - L7) - (L8 - L9)
        
        # Sequence 3: Top cue is w3
        L10 = get_L('00100', '00000')
        L11 = get_L('00100', '00010') # Drop effect of w4
        L12 = get_L('00100', '00011') # Drop effect of w5
        m3 = (L10 - L11) - (L11 - L12)
        
        subj_metrics.append(m1 + m2 + m3)
        
    return float(np.mean(subj_metrics))
outcome: self_sim=0.0102 (var=6.7556) adversary_sim=-0.3461 (var=5.5846) welch_t=+0.507 p=0.6144 (N=25, alpha=0.01) -> reject

[4] rationale: This metric directly operationalizes the primary dissociation in the experimental design: the effect of shared cues. The Sequential Evidence Accumulation theory calculates a running difference, meaning cues shared by both options cancel out to exactly zero and have no impact on the accumulated evidence or stopping probability. Thus, it predicts the choice probability will be identical between Trial 1 and Trial 4, and between Trial 5 and Trial 8 (expected difference = 0). In contrast, the Leaky Competing Accumulator theory processes evidence in parallel absolute terms. Adding shared cues increases the total input to both options' accumulators. Because of the zero-bound on accumulator activation and mutual inhibition, higher total baseline input can lift the losing option's accumulator away from zero, reducing the relative advantage of the winning option and systematically shifting choices toward the weaker option (increasing the response rate for B). By measuring the shift in response rate when shared cues are fully added, we isolate a strong positive signal for LCA while SEA remains tightly centered on zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    subj_diffs = []
    for subj, df in data.groupby('subject_id'):
        def get_r(a, b):
            mask = (df['A_str'] == a) & (df['B_str'] == b)
            if not mask.any():
                return 0.5
            return df[mask]['response'].mean()
            
        # Trial 1 vs Trial 4
        r1 = get_r('10000', '01000')
        r4 = get_r('10111', '01111')
        
        # Trial 5 vs Trial 8
        r5 = get_r('01000', '00100')
        r8 = get_r('11011', '10111')
        
        # Difference in response rate (P(B)) when shared cues are added
        diff = (r4 - r1) + (r8 - r5)
        subj_diffs.append(diff)
        
    return float(pd.Series(subj_diffs).mean())
outcome: self_sim=-0.0450 (var=0.0842) adversary_sim=-0.0183 (var=0.0812) welch_t=-0.328 p=0.7445 (N=25, alpha=0.01) -> reject

[5] rationale: The advocated Sequential Evidence Accumulation (SEA) theory calculates evidence as a running difference. Because shared cues produce a difference of exactly zero, they are completely ignored during accumulation. Consequently, SEA predicts that the underlying choice probability is mathematically identical across all trials within Group 1 (Trials 1-4) and Group 2 (Trials 5-8). Any variation in the empirical response rates across these trials is purely due to binomial noise. In contrast, the competing Leaky Competing Accumulator (LCA) theory evaluates absolute evidence in parallel. Adding shared cues increases the total input to the accumulators, which interacts with the leak and mutual inhibition dynamics to systematically alter the choice probabilities. By calculating the standard deviation of response rates within these trial groups, we capture the systematic variance induced by shared cues in LCA, whereas SEA will yield a tightly bounded, smaller value reflecting only random noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    g1_A = ['10000', '10100', '10110', '10111']
    g2_A = ['01000', '11000', '11010', '11011']
    
    subj_stds = []
    for subj, df in data.groupby('subject_id'):
        r1 = [df[df['A_str'] == a]['response'].mean() for a in g1_A if (df['A_str'] == a).any()]
        r2 = [df[df['A_str'] == a]['response'].mean() for a in g2_A if (df['A_str'] == a).any()]
        
        s1 = np.std(r1, ddof=1) if len(r1) > 1 else 0.0
        s2 = np.std(r2, ddof=1) if len(r2) > 1 else 0.0
        
        subj_stds.append(s1 + s2)
        
    return float(np.mean(subj_stds))
outcome: self_sim=0.2608 (var=0.0064) adversary_sim=0.2584 (var=0.0057) welch_t=+0.111 p=0.9123 (N=25, alpha=0.01) -> reject

[6] rationale: This metric calculates the difference in the mean error rate (Probability of choosing Option B) between Trials 5-8 and Trials 1-4. 

In the advocated Sequential Evidence Accumulation (SEA) theory, evidence is accumulated as a running difference, ignoring shared cues. Trials 1-4 always evaluate the difference between Cue 1 and Cue 2 (w1 - w2), while Trials 5-8 evaluate the difference between Cue 2 and Cue 3 (w2 - w3). Since w1 - w2 is generally greater than or equal to w2 - w3, SEA predicts that the evidence for Option A is stronger in Trials 1-4. Consequently, the error rate is lower in Trials 1-4, making the difference (Trials 5-8 error minus Trials 1-4 error) strictly POSITIVE.

In the competing Leaky Competing Accumulator (LCA) theory, absolute evidence is accumulated in parallel. Shared cues significantly boost the total input to the accumulators. In Trials 5-8, the strongest cue (Cue 1) is shared in 3 out of 4 trials, massively increasing the total input to Option A's accumulator. When mutual inhibition is strong, this higher total input drives Option A's activation higher than in Trials 1-4 (where Cue 1 is only present for Option A and not shared). Because of this input boost, LCA predicts that Option A will dominate even more in Trials 5-8, leading to a lower error rate. Thus, LCA predicts this metric to be NEGATIVE or near zero, structurally dissociating it from SEA.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    g1 = ['10000', '10100', '10110', '10111']
    g2 = ['01000', '11000', '11010', '11011']
    
    diffs = []
    for subj, df in data.groupby('subject_id'):
        r1 = df[df['A_str'].isin(g1)]['response'].mean()
        r2 = df[df['A_str'].isin(g2)]['response'].mean()
        diffs.append(r2 - r1)
        
    return float(pd.Series(diffs).mean())
outcome: self_sim=0.0400 (var=0.0094) adversary_sim=-0.0029 (var=0.0090) welch_t=+1.583 p=0.1201 (N=25, alpha=0.01) -> reject

[7] rationale: This metric builds upon the dissociation between the models' handling of shared cues and stopping rules, but introduces a continuous performance weighting to suppress binomial noise from high-lapse-rate subjects. 

In the advocated Sequential Evidence Accumulation (SEA) theory, evidence is accumulated as a running difference, ignoring shared cues. Trials 1-4 (Group 1) evaluate w1 - w2, while Trials 5-8 (Group 2) evaluate w2 - w3. Because w1 > w2, SEA often triggers its early stopping rule in Group 1, leading to a very high probability of choosing A. In Group 2, the evidence is weaker, leading to a higher error rate. Thus, SEA predicts r2 > r1, making the difference strictly positive.

In the competing Leaky Competing Accumulator (LCA) theory, absolute evidence is accumulated in parallel. Shared cues boost the total input. In Group 2, the strongest cue (w1) is shared in 3 out of 4 trials, massively increasing total input. Due to LCA's mutual inhibition and zero-bounded accumulators, this higher baseline input lifts the losing option's accumulator into the linear regime, which paradoxically increases the relative advantage of Option A, leading to a lower error rate in Group 2. Thus, LCA predicts r2 < r1, making the difference negative.

By multiplying this difference by `max(0, accuracy - 0.5)`, we smoothly zero-out the binomial noise from subjects who are guessing randomly (due to high epsilon/temperature), tightly bounding the between-subject variance while preserving the strong positive mean for SEA and the negative/zero mean for LCA.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data = data.copy()
    # Project list columns to string for hashability
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Group 1: Trials 1-4 (Differentiating cues w1 vs w2, no shared top cue)
    g1 = ['10000', '10100', '10110', '10111']
    # Group 2: Trials 5-8 (Differentiating cues w2 vs w3, with massive shared cues including top cue w1)
    g2 = ['01000', '11000', '11010', '11011']
    
    subj_metrics = []
    for subj, df in data.groupby('subject_id'):
        # Calculate overall accuracy (probability of choosing the objectively better Option A)
        acc = 1.0 - df['response'].mean()
        
        # Continuous weighting to down-weight subjects with high lapse rates (near-random choices).
        # This drastically reduces between-subject binomial noise without introducing hard thresholds.
        weight = max(0.0, acc - 0.5)
        
        df_g1 = df[df['A_str'].isin(g1)]
        df_g2 = df[df['A_str'].isin(g2)]
        
        if len(df_g1) == 0 or len(df_g2) == 0:
            subj_metrics.append(0.0)
            continue
            
        # r1 and r2 are the error rates (probability of choosing B) for each group
        r1 = df_g1['response'].mean()
        r2 = df_g2['response'].mean()
        
        # Difference in error rates
        diff = r2 - r1
        
        subj_metrics.append(diff * weight)
        
    if not subj_metrics:
        return 0.0
    return float(pd.Series(subj_metrics).mean())
outcome: self_sim=0.0016 (var=0.0001) adversary_sim=-0.0002 (var=0.0001) welch_t=+0.566 p=0.5742 (N=25, alpha=0.01) -> reject

[8] rationale: This metric exploits the interaction between early stopping and cue evaluation order in the Sequential Evidence Accumulation (SEA) model. In SEA, cues are evaluated sequentially in descending order of validity. When two cues have identical validity (e.g., w3 and w4, or w5 and w6), the stable sort evaluates the earlier-indexed cue first. In Trial 2 and 5, Option A possesses the earlier-evaluated cue of the tied pair, causing the running evidence to temporarily spike towards Option A, increasing the probability of crossing the early-stopping threshold and choosing A. In Trial 3 and 6, Option B possesses the earlier-evaluated cue, causing the running evidence to temporarily dip towards Option B, increasing the probability of early-stopping on B. Thus, SEA predicts P(A) is higher in Trials 2 and 5 compared to Trials 3 and 6. In contrast, the Leaky Competing Accumulator (LCA) model processes all cues in parallel. Because the tied cues have identical validity, the total parallel input to Option A and Option B is mathematically identical between Trial 2 and 3, and between Trial 5 and 6. LCA therefore predicts an expected difference of exactly zero. This provides a clean, structural dissociation between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    subj_diffs = []
    for subj, df in data.groupby('subject_id'):
        def get_pA(a_str):
            mask = df['A_str'] == a_str
            if not mask.any(): return 0.5
            return 1.0 - df[mask]['response'].mean()
            
        # Trial 2 vs Trial 3
        r2 = get_pA('101000')
        r3 = get_pA('100100')
        
        # Trial 5 vs Trial 6
        r5 = get_pA('101010')
        r6 = get_pA('100101')
        
        subj_diffs.append((r2 - r3) + (r5 - r6))
        
    return float(pd.Series(subj_diffs).mean())
outcome: self_sim=-0.0300 (var=0.0338) adversary_sim=0.0063 (var=0.0666) welch_t=-0.572 p=0.5702 (N=25, alpha=0.01) -> reject

[9] rationale: This metric leverages a mathematical impossibility in the advocated Sequential Evidence Accumulation (SEA) model to perfectly dissociate it from the Leaky Competing Accumulator (LCA). In SEA, cues are evaluated sequentially in descending order of validity. Because w0 is strictly greater than w1, and w1 > w2, the accumulated evidence reaches its absolute maximum at step 0 (E = w0). All subsequent peaks in the evidence trajectory (like w0 - w1 + w2) are mathematically guaranteed to be smaller than w0. Consequently, if SEA does not cross its stopping threshold at step 0, it will NEVER cross it at any subsequent step, meaning early stopping on later cues is impossible. Thus, SEA predicts that the choice probability P(A) is exactly identical across Trial 1, Trial 5, and Trial 6, yielding an expected difference of exactly zero.

In contrast, the LCA model evaluates all cues in parallel. Trials 5 and 6 contain multiple shared lower-validity cues that are absent in Trial 1, drastically increasing the total baseline input to the accumulators. Due to LCA's zero-bounding and mutual inhibition, this higher total input amplifies the winner-take-all dynamics, systematically increasing the probability of choosing the leading option (Option A). By calculating P(A | Trials 5,6) - P(A | Trial 1), we isolate a strictly positive shift predicted by LCA against a tightly bounded zero prediction from SEA.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # High total input trials (Trials 5 and 6)
    high_cues = ['101010', '100101']
    # Low total input trial (Trial 1)
    low_cues = ['100000']
    
    subj_diffs = []
    for subj, df in data.groupby('subject_id'):
        df_high = df[df['A_str'].isin(high_cues)]
        df_low = df[df['A_str'].isin(low_cues)]
        
        if len(df_high) == 0 or len(df_low) == 0:
            continue
            
        # Probability of choosing Option A
        pA_high = 1.0 - df_high['response'].mean()
        pA_low = 1.0 - df_low['response'].mean()
        
        subj_diffs.append(pA_high - pA_low)
        
    if not subj_diffs:
        return 0.0
    return float(pd.Series(subj_diffs).mean())
outcome: self_sim=-0.0031 (var=0.0179) adversary_sim=-0.0075 (var=0.0167) welch_t=+0.118 p=0.9068 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    return float(data['response'].mean())",
  "rationale": "In this specific experimental design, the difference in validities between the best cue for Option A (0.9) and Option B (0.85) is small, and all subsequent added cues are perfectly balanced. The Sequential Evidence Accumulation (SEA) model evaluates this difference exactly once (w0 - w1); because this difference is small and the threshold theta is rarely crossed at step 0, the final accumulated evidence is weak, leading to a choice probability closer to 0.5. In contrast, the Leaky Competing Accumulator (LCA) integrates this evidence over 50 time steps (total time 5.0). This temporal integration, combined with mutual inhibition, massively amplifies the small w0 - w1 advantage. Consequently, LCA behaves much more deterministically, choosing Option A almost always (bounded only by the lapse rate epsilon). Therefore, the overall error rate (the probability of choosing Option B, which is exactly the mean of the 'response' column) will be systematically and significantly higher in SEA than in LCA."
}
```

## Usage

```json
{
  "prompt_token_count": 10395,
  "candidates_token_count": 254,
  "total_token_count": 21947
}
```
