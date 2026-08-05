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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=9):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design provides a strict, parameter-free qualitative dissociation between the Advocated model (TTB with Sub-linear Tallying) and the Competing model (Weighted Additive with Power-Function Validity Scaling). The Advocated model assigns a fixed weight (beta_ttb) to the highest-validity discriminating cue, completely ignoring its absolute validity magnitude, and then tallies the remaining cues. By holding the number of remaining cues won by Option B constant (e.g., B always wins the 5th cue) while systematically shifting the identity of Option A's best cue from the 1st to the 4th cue across trials, the Advocated model is mathematically forced to predict identical choice probabilities across these trials. In stark contrast, the Competing model scales the evidence of each cue by a power function of its actual validity. As Option A's single winning cue drops in validity across trials, the Competing model predicts a systematic decrease in the probability of choosing Option A.

**Computed schedule:** 9 unique pairs × 10 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Take-The-Best with Compensatory Verification (Sub-linear Tallying)

**Parameters:**
- beta_ttb: [0.0, 10.0]
- beta_tally: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    ttb_winner = -1
    disc_idx = -1
    for i, j in enumerate(cue_order):
        if a[j] != b[j]:
            ttb_winner = 0 if a[j] > b[j] else 1
            disc_idx = i
            break
            
    if ttb_winner == -1:
        return np.array([0.5, 0.5])
        
    # Tally remaining cues (those evaluated after the discriminating cue)
    remaining_a = 0.0
    remaining_b = 0.0
    for j in cue_order[disc_idx+1:]:
        remaining_a += a[j]
        remaining_b += b[j]
        
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    
    tally_diff = remaining_a - remaining_b
    compressed_diff = np.sign(tally_diff) * np.sqrt(np.abs(tally_diff))
    
    score_a = beta_ttb * (1.0 if ttb_winner == 0 else 0.0) + beta_tally * compressed_diff
    score_b = beta_ttb * (1.0 if ttb_winner == 1 else 0.0)
    
    scores = np.array([score_a, score_b])
    z = scores - np.max(scores)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
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
**Description:** Weighted Additive with Power-Function Validity Scaling (WADD-PWR): Decision-makers integrate all available cues but weight them non-linearly using a power function of their stated validities. A single parameter, gamma, controls the weighting scheme: positive values approximate Take-The-Best by heavily favoring the most valid cues, zero yields Tallying by weighting all cues equally, and negative values approximate Reverse TTB by favoring less valid cues. The power function naturally anchors a validity of 1.0 to a weight of 1.0, providing built-in numerical stability across the full spectrum of strategies without requiring artificial normalization.

**Parameters:**
- gamma: [-20.0, 20.0]
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
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Apply power-function weighting scheme to capture TTB, Tallying, and Reverse TTB
    # Validities are in [0.5, 1.0], so val ** gamma is numerically stable.
    w = val ** gamma
    
    # Accumulate evidence based on discriminating cues
    diff = a - b
    ev_a = np.sum(w[diff > 0])
    ev_b = np.sum(w[diff < 0])
    
    # Convert to choice probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * scores
    e = np.exp(z - np.max(z))
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
[0] rationale: The Advocated theory (Take-The-Best with Sub-linear Tallying) predicts identical choice probabilities for specific pairs of trials where the TTB winner and the tally difference of the remaining cues are strictly held constant. By measuring the absolute difference in the empirical probability of choosing option A within these matched pairs, we expect a value close to 0 (driven only by binomial noise) for the Advocated theory. In contrast, the Competing theory (WADD-PWR) weights cues non-linearly by their validities, meaning it predicts systematically shifting choice probabilities across these pairs depending on the specific lower-validity cues involved. This metric aggregates the absolute differences across the three matched pairs, which should be substantially larger for the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_key'] = data['option_a_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    p_a = data.groupby('A_key')['response'].apply(lambda x: (x == 0).mean())
    
    diff = 0.0
    pairs = [
        ("10100", "10001"),
        ("10110", "10011"),
        ("11010", "11001")
    ]
    
    for k1, k2 in pairs:
        if k1 in p_a.index and k2 in p_a.index:
            diff += abs(p_a[k1] - p_a[k2])
            
    return float(diff)
outcome: self_sim=0.0350 (var=0.0227) adversary_sim=0.0587 (var=0.0263) welch_t=-0.537 p=0.594 (N=25, alpha=0.01) -> reject

[1] rationale: In the Advocated theory (TTB + Sub-linear Tallying), the choice probability is driven by the validity rank of the discriminating cue and the tally difference of the remaining cues. For trials 1 and 2, the discriminating cue is the 1st cue (validity 0.95), and the tally difference of the remaining cues is 0. For trials 5 and 6, the discriminating cue is the 2nd cue (validity 0.85), and the tally difference is also 0. Crucially, the Advocated theory does not scale the TTB winner's weight by its validity—it simply assigns a constant `beta_ttb` to whichever cue wins. Thus, it predicts EXACTLY the same choice probability for options A in trials 1 and 2 as in trials 5 and 6. In contrast, the Competing theory (WADD-PWR) weights cues non-linearly by their validities, meaning the evidence difference heavily depends on whether the 1st or 2nd cue is the best discriminating cue. By taking the squared difference between the empirical probability of choosing A in {T1, T2} and {T5, T6} per subject, we expect a value close to 0 (driven only by binomial variance) for the Advocated theory, and a substantially larger value for the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_key'] = data['option_a_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    
    def subject_score(df):
        is_a = (df['response'] == 0)
        t12 = df['A_key'].isin(["10100", "10001"])
        t56 = df['A_key'].isin(["11010", "11001"])
        
        p12 = is_a[t12].mean() if t12.sum() > 0 else 0.5
        p56 = is_a[t56].mean() if t56.sum() > 0 else 0.5
        
        return (p12 - p56) ** 2

    scores = data.groupby('subject_id').apply(subject_score)
    return float(scores.mean())
outcome: self_sim=0.0092 (var=0.0002) adversary_sim=0.0197 (var=0.0015) welch_t=-1.278 p=0.2109 (N=25, alpha=0.01) -> reject

[2] rationale: To avoid the positive bias of absolute differences on noisy empirical probabilities, we compute the product of differences across pairs. In the Advocated theory, the true difference in choice probability for each matched pair (T1 vs T2, T3 vs T4, T5 vs T6) is exactly zero. Because the empirical differences are independent mean-zero noise, their products strongly regress to zero in expectation. In the Competing theory (WADD-PWR), cues are weighted non-linearly. T1, T3, and T5 contain stronger secondary cues than T2, T4, and T6, respectively. Consequently, the differences (T1-T2), (T3-T4), and (T5-T6) will all share the exact same sign (positive if gamma > 0, negative if gamma < 0). Their pairwise products will therefore be systematically positive, providing a clean separation between the theories without zero-inflation.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_key'] = data['option_a_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    p_a = data.groupby('A_key')['response'].apply(lambda x: (x == 0).mean())
    
    def get_p(k):
        return p_a.get(k, 0.5)
        
    d12 = get_p("10100") - get_p("10001")
    d34 = get_p("10110") - get_p("10011")
    d56 = get_p("11010") - get_p("11001")
    
    return float(d12 * d34 + d12 * d56 + d34 * d56)
outcome: self_sim=0.0001 (var=0.0005) adversary_sim=0.0010 (var=0.0010) welch_t=-0.117 p=0.9073 (N=25, alpha=0.01) -> reject

[3] rationale: The Advocated theory (TTB + Sub-linear Tallying) predicts identical choice probabilities for the matched pairs (T1 vs T2), (T3 vs T4), and (T5 vs T6) because the TTB winner and the tally difference of the remaining cues are strictly held constant within each pair. Thus, the expected difference in choice probabilities for each pair is exactly zero, and the sum of these differences will just be binomial noise. The Competing theory (WADD-PWR) weights cues non-linearly by their validities. Because the 'high' trials in each pair always feature a stronger secondary cue than the 'low' trials, the evidence differences will all share the exact same sign (positive if gamma > 0, negative if gamma < 0). By summing these three differences per subject and squaring the result, we capture this systematic, uni-directional shift while allowing the noise in the Advocated theory to average out before squaring.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_key'] = data['option_a_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    
    def subject_score(df):
        p = df.groupby('A_key')['response'].apply(lambda x: (x == 0).mean())
        d1 = p.get("10100", 0.5) - p.get("10001", 0.5)
        d2 = p.get("10110", 0.5) - p.get("10011", 0.5)
        d3 = p.get("11010", 0.5) - p.get("11001", 0.5)
        return (d1 + d2 + d3)**2

    scores = data.groupby('subject_id').apply(subject_score)
    return float(scores.mean())
outcome: self_sim=0.0377 (var=0.0022) adversary_sim=0.0471 (var=0.0065) welch_t=-0.506 p=0.6154 (N=25, alpha=0.01) -> reject

[4] rationale: This metric exploits the structural difference in evidence accumulation between the two theories. We isolate trials 3, 4, and 7, where Option A's features are held constant ('10000') and Option B wins 0, 1, or 4 lower-validity cues, respectively. In the Competing theory (WADD-PWR), evidence changes linearly with the number of cues, so the drop in log-odds from T3 to T4 is only 1/4th of the drop from T3 to T7. In contrast, the Advocated theory (TTB + Sub-linear Tallying) applies a square-root transformation to the tally difference. Because sqrt(1) - sqrt(0) = 1 and sqrt(4) - sqrt(1) = 1, the evidence drop from T3 to T4 is exactly equal to the evidence drop from T4 to T7. By computing the contrast (P3 - P4) - (P4 - P7) = P3 + P7 - 2*P4, we expect a value near zero for the Advocated theory, and a strongly negative value for the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['B_key'] = data['option_b_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    data['A_key'] = data['option_a_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    
    df = data[data['A_key'] == '10000']
    
    def get_p(df_sub):
        p3_series = df_sub[df_sub['B_key'] == '00000']['response']
        p4_series = df_sub[df_sub['B_key'] == '01000']['response']
        p7_series = df_sub[df_sub['B_key'] == '01111']['response']
        
        p3 = (p3_series == 0).mean() if len(p3_series) > 0 else 0.5
        p4 = (p4_series == 0).mean() if len(p4_series) > 0 else 0.5
        p7 = (p7_series == 0).mean() if len(p7_series) > 0 else 0.5
        
        return p3 + p7 - 2 * p4

    if 'subject_id' in data.columns and data['subject_id'].nunique() > 1:
        return float(df.groupby('subject_id').apply(get_p).mean())
    else:
        return float(get_p(df))
outcome: self_sim=0.2917 (var=0.2406) adversary_sim=0.3783 (var=0.2914) welch_t=-0.594 p=0.5553 (N=25, alpha=0.01) -> reject

[5] rationale: This metric exploits the stark difference in evidence accumulation between the sub-linear Tallying of the Advocated theory and the linear accumulation of the Competing theory (WADD-PWR). We compare the empirical probability of choosing Option A across three key trials: T2 (net +1 lower cue for A), T4 (net -1 lower cue for A), and T7 (net -4 lower cues for A). In the Advocated theory, the tally difference undergoes a square-root transformation, meaning the evidence drop from T2 to T4 (a 2-unit raw difference) is substantially larger than the drop from T4 to T7 (a 3-unit raw difference but heavily compressed). Thus, P(T2) - P(T4) > P(T4) - P(T7), making the contrast `2*P(T4) - P(T2) - P(T7)` strongly negative. Conversely, in the Competing theory, evidence changes linearly with the number of cues. The evidence drop from T4 to T7 is strictly 1.5 times larger than the drop from T2 to T4, leading to P(T4) - P(T7) > P(T2) - P(T4), and making the contrast strongly positive.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_key'] = data['option_a_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    data['B_key'] = data['option_b_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    
    def get_p(df_sub):
        t2 = df_sub[(df_sub['A_key'] == '11000') & (df_sub['B_key'] == '00000')]
        t4 = df_sub[(df_sub['A_key'] == '10000') & (df_sub['B_key'] == '01000')]
        t7 = df_sub[(df_sub['A_key'] == '10000') & (df_sub['B_key'] == '01111')]
        
        p2 = (t2['response'] == 0).mean() if len(t2) > 0 else 0.5
        p4 = (t4['response'] == 0).mean() if len(t4) > 0 else 0.5
        p7 = (t7['response'] == 0).mean() if len(t7) > 0 else 0.5
        
        return 2 * p4 - p2 - p7

    if 'subject_id' in data.columns and data['subject_id'].nunique() > 1:
        return float(data.groupby('subject_id').apply(get_p).mean())
    else:
        return float(get_p(data))
outcome: self_sim=-0.3550 (var=0.2497) adversary_sim=-0.3767 (var=0.3278) welch_t=+0.143 p=0.8872 (N=25, alpha=0.01) -> reject

[6] rationale: This metric exploits the structural difference in evidence accumulation between the two theories by identifying their respective 'midpoint' trials. We examine trials 3, 4, 5, and 7, where Option A's features are held constant and Option B wins 0, 1, 2, or 4 lower-validity cues. In the Competing theory (WADD-PWR), evidence accumulates linearly, meaning the evidence for Trial 5 (B wins 2 cues) is exactly halfway between Trial 3 (B wins 0) and Trial 7 (B wins 4). In contrast, the Advocated theory (TTB + Sub-linear Tallying) applies a square-root transformation to the tally difference. Because sqrt(1) is exactly halfway between sqrt(0) and sqrt(4), Trial 4 (B wins 1 cue) is the true evidence midpoint for the Advocated theory. We compute the absolute deviation of P(Trial 5) and P(Trial 4) from the empirical midpoint (P3 + P7) / 2. The contrast 'dev_5 - dev_4' will be strictly positive for the Advocated theory (since T4 is the true midpoint) and strictly negative for the Competing theory (since T5 is the true midpoint).
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_key'] = data['option_a_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    data['B_key'] = data['option_b_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    
    def subject_score(df_sub):
        p3_series = df_sub[(df_sub['A_key'] == '10000') & (df_sub['B_key'] == '00000')]['response']
        p4_series = df_sub[(df_sub['A_key'] == '10000') & (df_sub['B_key'] == '01000')]['response']
        p5_series = df_sub[(df_sub['A_key'] == '10000') & (df_sub['B_key'] == '01100')]['response']
        p7_series = df_sub[(df_sub['A_key'] == '10000') & (df_sub['B_key'] == '01111')]['response']
        
        p3 = (p3_series == 0).mean() if len(p3_series) > 0 else 0.5
        p4 = (p4_series == 0).mean() if len(p4_series) > 0 else 0.5
        p5 = (p5_series == 0).mean() if len(p5_series) > 0 else 0.5
        p7 = (p7_series == 0).mean() if len(p7_series) > 0 else 0.5
        
        dev_5 = abs(2 * p5 - p3 - p7)
        dev_4 = abs(2 * p4 - p3 - p7)
        
        return dev_5 - dev_4

    if 'subject_id' in data.columns and data['subject_id'].nunique() > 1:
        return float(data.groupby('subject_id').apply(subject_score).mean())
    else:
        return float(subject_score(data))
outcome: self_sim=0.0500 (var=0.0881) adversary_sim=0.0167 (var=0.0858) welch_t=+0.400 p=0.6912 (N=25, alpha=0.01) -> reject

[7] rationale: This metric evaluates the linearity of evidence accumulation by computing the empirical log-odds of choosing Option A. In the Competing theory (WADD-PWR), evidence changes linearly with the number of cues Option B wins. Because the softmax function maps evidence to probabilities via the logistic function, the true log-odds are strictly linear. Thus, the contrast L(T3) + L(T7) - 2*L(T5) is theoretically exactly 0 for WADD-PWR, regardless of its parameters. In contrast, the Advocated theory applies a square-root transformation to the tally difference, meaning the evidence drops sub-linearly. Calculating the same log-odds contrast for the Advocated theory yields a systematically positive value (proportional to 2*sqrt(2) - 2 > 0). Using the Haldane-Anscombe correction ensures robust, bounded log-odds estimates per subject.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    data = data.copy()
    data['A_key'] = data['option_a_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    data['B_key'] = data['option_b_ratings'].apply(lambda x: "".join(str(int(v)) for v in x))
    
    df = data[data['A_key'] == '10000']
    
    def subject_score(df_sub):
        t3 = df_sub[df_sub['B_key'] == '00000']
        t5 = df_sub[df_sub['B_key'] == '01100']
        t7 = df_sub[df_sub['B_key'] == '01111']
        
        k3 = (t3['response'] == 0).sum()
        k5 = (t5['response'] == 0).sum()
        k7 = (t7['response'] == 0).sum()
        
        n3 = len(t3) if len(t3) > 0 else 12
        n5 = len(t5) if len(t5) > 0 else 12
        n7 = len(t7) if len(t7) > 0 else 12
        
        # Haldane-Anscombe correction to estimate log-odds without infinity
        L3 = np.log((k3 + 0.5) / (n3 - k3 + 0.5))
        L5 = np.log((k5 + 0.5) / (n5 - k5 + 0.5))
        L7 = np.log((k7 + 0.5) / (n7 - k7 + 0.5))
        
        # Contrast: L3 + L7 - 2*L5
        return L3 + L7 - 2 * L5

    if 'subject_id' in data.columns and data['subject_id'].nunique() > 1:
        scores = df.groupby('subject_id').apply(subject_score)
        return float(scores.mean())
    else:
        return float(subject_score(df))
outcome: self_sim=2.3850 (var=4.5471) adversary_sim=2.6934 (var=8.0022) welch_t=-0.435 p=0.6654 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['A_key'] = data['option_a_ratings'].apply(lambda x: \"\".join(str(int(v)) for v in x))\n    data['B_key'] = data['option_b_ratings'].apply(lambda x: \"\".join(str(int(v)) for v in x))\n    \n    def get_excess_var(df_sub, b_key):\n        df_b = df_sub[df_sub['B_key'] == b_key]\n        if len(df_b) < 2:\n            return 0.0\n        \n        agg = df_b.groupby('A_key')['response'].agg(['sum', 'count'])\n        # response == 0 means option A was chosen\n        agg['p_A'] = 1.0 - (agg['sum'] / agg['count'])\n        \n        if len(agg) < 2:\n            return 0.0\n            \n        total_A = (agg['count'] - agg['sum']).sum()\n        total_count = agg['count'].sum()\n        p_mean = total_A / total_count\n        \n        # Empirical variance of choice probabilities across different A options\n        v = agg['p_A'].var(ddof=1)\n        n_mean = agg['count'].mean()\n        \n        # Expected binomial variance if true probability is constant\n        expected_v = p_mean * (1.0 - p_mean) / n_mean\n        \n        return v - expected_v\n\n    def subject_score(df_sub):\n        # Group 1: Trials 1-4 (B is always 00001)\n        ev1 = get_excess_var(df_sub, '00001')\n        # Group 2: Trials 5-7 (B is always 00011)\n        ev2 = get_excess_var(df_sub, '00011')\n        return ev1 + ev2\n\n    if 'subject_id' in data.columns and data['subject_id'].nunique() > 0:\n        return float(data.groupby('subject_id').apply(subject_score).mean())\n    else:\n        return float(subject_score(data))",
  "rationale": "This metric calculates the 'excess variance' of choice probabilities across trials that share the same Option B and where Option A has exactly one winning cue. In the Advocated theory (TTB + Sub-linear Tallying), because the TTB rank and tally difference are strictly held constant within these trial groups (Trials 1-4, and Trials 5-7), the true probability of choosing Option A is identical across them. Thus, the empirical variance of choice probabilities should perfectly match the expected binomial variance, yielding an excess variance of zero. In contrast, the Competing theory (WADD-PWR) scales evidence non-linearly by the exact validities of the cues. As Option A's winning cue shifts from the 1st to the 4th cue, the true choice probabilities systematically vary, resulting in a large, strictly positive excess variance. By computing this excess variance per subject, we cleanly separate the theories regardless of whether the Competing theory's parameter (gamma) is positive or negative, avoiding any cancellation effects across subjects."
}
```

## Usage

```json
{
  "prompt_token_count": 8720,
  "candidates_token_count": 796,
  "total_token_count": 15427
}
```
