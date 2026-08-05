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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate Contrarian WADD from TTB with High Noise, we must exploit the 'alpha' parameter in Contrarian WADD, which can be negative. A negative alpha implies an 'Anti-Tallying' or contrarian preference for the option with the *lower* weighted sum of features. TTB, regardless of its high noise, will always consistently predict a weak preference for the option that wins on the highest-validity discriminating cue. By designing trials where the TTB-favored option sometimes has a higher weighted sum and sometimes has a lower weighted sum, we can strictly dissociate the models. If subjects consistently choose the option with the lower weighted sum (even when it loses on the top cue, or is completely dominated), it is exclusively predicted by Contrarian WADD and strictly falsifies TTB.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Contrarian WADD: Decision makers evaluate options by computing a weighted sum of their features, but they may distrust the provided expert ratings or view them as added complexity. Thus, they apply a scaling factor to the validities that can be negative, leading to an 'Anti-Tallying' or contrarian preference for options with lower scores. This weak contrarian signal is obscured by a very high rate of random guessing (lapse rate).

**Parameters:**
- alpha: [-2.0, 1.0]
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Contrarian WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Compute scores with the alpha scaling factor (which can be negative)
    score_a = np.dot(stim[0], val) * alpha
    score_b = np.dot(stim[1], val) * alpha
    scores = np.array([score_a, score_b])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Take-The-Best (TTB) with High Noise: Decision makers employ a lexicographic heuristic, searching through cues in order of descending validity. They stop at the first cue that discriminates between the two options and choose the option with the higher value on that cue. However, to accommodate the empirical observation that agreement with any deterministic strategy hovers around 50%, the model incorporates a very high lapse rate (epsilon) and a low softmax inverse temperature (beta). This restricts the model to primarily exhibit random guessing, with only a weak TTB signal, matching the high degree of noise in the observed data.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity; stable sort handles ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    
    # Lexicographic search
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No cue discriminates, guess uniformly
        return np.array([0.5, 0.5])
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over binary TTB scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
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
[0] rationale: This metric computes the within-subject Pearson correlation between the TTB model's binary prediction for Option A and the subject's empirical probability of choosing Option A across the 8 unique trial types.

For the 'TTB with High Noise' model, the empirical probability of choosing A will reliably be higher when TTB predicts A (due to the weak lexicographic signal), resulting in a consistently positive correlation (around +0.4 to +0.5).

For the 'Contrarian WADD' model, the alpha parameter is predominantly negative (mean -0.5), meaning the model generally prefers the option with the LOWER weighted sum of features. Because the TTB prediction is highly positively correlated with the weighted sum, Contrarian WADD's choice probabilities will generally move in the opposite direction of the TTB prediction. This results in a negative correlation on average (around -0.2 to -0.3). The substantial difference in the mean correlation (positive vs. negative) provides highly reliable discrimination between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Convert binary rating arrays to string keys to safely identify trial types
    data['trial_type'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, x))) + "_" + data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    # Map each trial type to the TTB model's prediction for Option A (1 if TTB predicts A, 0 if B)
    # Lexicographic search favors the option with '1' on the first discriminating cue.
    ttb_preds = {
        "1111_0000": 1,
        "1000_0111": 1,
        "0111_1000": 0,
        "1001_0110": 1,
        "0000_1111": 0,
        "1100_0011": 1,
        "0011_1100": 0,
        "0100_0011": 1
    }
    
    df = data.copy()
    df['ttb_pred_A'] = df['trial_type'].map(ttb_preds).fillna(0.5)
    df['chose_A'] = (df['response'] == 0).astype(float)
    
    # Calculate the empirical probability of choosing A per trial type per subject
    p_A_df = df.groupby(['subject_id', 'trial_type', 'ttb_pred_A'])['chose_A'].mean().reset_index()
    
    # Calculate Pearson correlation between TTB's prediction and the subject's choice probability
    def calc_corr(subj_data):
        if len(subj_data) < 2:
            return 0.0
        c = subj_data['ttb_pred_A'].corr(subj_data['chose_A'])
        return 0.0 if pd.isna(c) else c
        
    corrs = p_A_df.groupby('subject_id').apply(calc_corr)
    
    # Return the mean correlation across all subjects
    return float(corrs.mean())
outcome: self_sim=-0.0567 (var=0.1099) adversary_sim=0.0424 (var=0.1683) welch_t=-0.939 p=0.3525 (N=25, alpha=0.01) -> reject

[1] rationale: This metric leverages the structural disagreement between TTB and WADD across two carefully selected pairs of mirrored trials. 

Let dX be the difference in the probability of choosing A between Trial 1 (A dominates) and Trial 5 (B dominates). For TTB, dX is positive. For WADD, dX is proportional to +6.4 * alpha.

Let dY be the difference in the probability of choosing A between Trial 2 (A has top cue, B has rest) and Trial 3 (B has top cue, A has rest). For TTB, the top cue dominates, so dY is positive. For WADD, the weighted sum favors the option with more cues, so dY is proportional to -2.6 * alpha.

The metric computes the product dX * dY per subject. 
For TTB, both dX and dY are positive, so their product is consistently positive.
For WADD, dX and dY always have opposite signs (since one is proportional to +alpha and the other to -alpha). Thus, their product is proportional to -16.64 * alpha^2, which is strictly negative regardless of whether the subject's alpha is positive or negative. This elegantly factors out the variance of the unknown alpha parameter and cleanly separates the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    df = data.copy()
    
    # Convert binary rating arrays to string keys to safely identify trial types
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x])) + "_" + df['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    df['chose_A'] = (df['response'] == 0).astype(float)
    
    def subject_score(subj_data):
        p_A = subj_data.groupby('trial_type')['chose_A'].mean()
        
        # Trial pair 1: Dominance
        # T1: A dominates B. TTB predicts A. WADD diff A-B = +3.2
        p_T1 = p_A.get("1111_0000", 0.5)
        # T5: B dominates A. TTB predicts B. WADD diff A-B = -3.2
        p_T5 = p_A.get("0000_1111", 0.5)
        dX = p_T1 - p_T5
        
        # Trial pair 2: Conflicting cues
        # T2: A=[1,0,0,0], B=[0,1,1,1]. TTB predicts A. WADD diff A-B = -1.3
        p_T2 = p_A.get("1000_0111", 0.5)
        # T3: A=[0,1,1,1], B=[1,0,0,0]. TTB predicts B. WADD diff A-B = +1.3
        p_T3 = p_A.get("0111_1000", 0.5)
        dY = p_T2 - p_T3
        
        # Return the product of the two differences
        return dX * dY

    scores = df.groupby('subject_id').apply(subject_score)
    return float(scores.mean())

outcome: self_sim=0.0028 (var=0.0014) adversary_sim=0.0065 (var=0.0024) welch_t=-0.307 p=0.7603 (N=25, alpha=0.01) -> reject

[2] rationale: This metric leverages a structural symmetry that perfectly isolates WADD from TTB regardless of WADD's unknown `alpha` parameter. 

We define three pairs of mirrored trials: (T1, T5), (T2, T3), and (T6, T7). For the Take-The-Best (TTB) model, the first cue identically discriminates all three pairs (predicting A for the first trial in the pair, and B for the second). Thus, under TTB, the differences in the empirical probability of choosing A for these pairs (X1, X2, and X3) are exactly identically distributed. Consequently, the expected value of `|X1| - 0.5 * (|X2| + |X3|)` is rigorously zero for TTB, even in the presence of high noise and small-sample bias.

For Contrarian WADD, however, the absolute differences in the underlying weighted sums vary drastically across these pairs: 6.4 for X1, 2.6 for X2, and 0.8 for X3. Because the choice probabilities are a function of these absolute differences, `|X1|` will reliably be much larger than `|X2|` and `|X3|`, regardless of whether `alpha` is positive or negative. Taking the absolute values prevents the metric from averaging out to zero across subjects with different alpha signs, yielding a strictly positive, highly robust signal for WADD and a tight zero for TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    df = data.copy()
    
    # Convert binary rating arrays to string keys to safely identify trial types
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x])) + "_" + df['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    df['chose_A'] = (df['response'] == 0).astype(float)
    
    def subject_score(subj_data):
        p_A = subj_data.groupby('trial_type')['chose_A'].mean()
        
        # X1: Dominance trials (T1 vs T5)
        # WADD score difference is massive (A-B = +3.2 vs -3.2)
        p_T1 = p_A.get("1111_0000", 0.5)
        p_T5 = p_A.get("0000_1111", 0.5)
        X1 = p_T1 - p_T5
        
        # X2: Conflicting cues (T2 vs T3)
        # WADD score difference is medium (A-B = -1.3 vs +1.3)
        p_T2 = p_A.get("1000_0111", 0.5)
        p_T3 = p_A.get("0111_1000", 0.5)
        X2 = p_T2 - p_T3
        
        # X3: Weak conflict (T6 vs T7)
        # WADD score difference is very small (A-B = +0.4 vs -0.4)
        p_T6 = p_A.get("1100_0011", 0.5)
        p_T7 = p_A.get("0011_1100", 0.5)
        X3 = p_T6 - p_T7
        
        # Return the contrast of absolute differences
        return abs(X1) - 0.5 * (abs(X2) + abs(X3))

    scores = df.groupby('subject_id').apply(subject_score)
    return float(scores.mean())

outcome: self_sim=0.0242 (var=0.0213) adversary_sim=-0.0467 (var=0.0196) welch_t=+1.752 p=0.08624 (N=25, alpha=0.01) -> reject

[3] rationale: This metric improves upon previous absolute-difference contrasts by massively reducing within-subject variance. Under TTB, the 8 trials perfectly cluster into two equivalence classes: 'predicts A' (T1, T2, T4, T6, T8) and 'predicts B' (T3, T5, T7). Within any class, or between any pair of classes, the absolute differences in choice probabilities are identically distributed for TTB. By exhaustively grouping these pairs into sets that have 'Large' vs 'Small' underlying differences under Contrarian WADD, we can take the mean of the Large set minus the mean of the Small set. For TTB, this contrast is perfectly balanced and has an expected value of exactly 0. For WADD, averaging over many pairs retains a strong positive signal while averaging out the binomial noise, yielding a highly significant t-statistic.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    df = data.copy()
    
    # Convert binary rating arrays to string keys to safely identify trial types
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x])) + "_" + df['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    df['chose_A'] = (df['response'] == 0).astype(float)
    
    def subject_score(subj_data):
        p = subj_data.groupby('trial_type')['chose_A'].mean()
        
        # Retrieve empirical probability of choosing A for each trial type
        T1 = p.get("1111_0000", 0.5)
        T2 = p.get("1000_0111", 0.5)
        T3 = p.get("0111_1000", 0.5)
        T4 = p.get("1001_0110", 0.5)
        T5 = p.get("0000_1111", 0.5)
        T6 = p.get("1100_0011", 0.5)
        T7 = p.get("0011_1100", 0.5)
        T8 = p.get("0100_0011", 0.5)
        
        # Under TTB, trials 1, 2, 4, 6, 8 all predict A with the exact same probability `p`.
        # Thus, any absolute difference between them has the exact same expected value.
        # We contrast pairs with LARGE WADD score differences vs SMALL WADD score differences.
        L_AA = [abs(T1 - T2), abs(T1 - T8), abs(T1 - T4), abs(T1 - T6), abs(T6 - T2)]
        S_AA = [abs(T6 - T4), abs(T4 - T8), abs(T8 - T2), abs(T6 - T8), abs(T4 - T2)]
        
        # Under TTB, trials 3, 5, 7 all predict B with the exact same probability `p`.
        L_BB = [abs(T3 - T5)]
        S_BB = [abs(T3 - T7)]
        
        # Under TTB, the absolute difference between any 'A' trial and any 'B' trial has the same expected value.
        L_AB = [abs(T1 - T5), abs(T6 - T5), abs(T1 - T7), abs(T4 - T5), abs(T8 - T5), abs(T2 - T3)]
        S_AB = [abs(T8 - T7), abs(T4 - T7), abs(T6 - T7), abs(T2 - T7), abs(T6 - T3), abs(T4 - T3)]
        
        # Compute the contrast between Large and Small WADD difference pairs
        score_AA = np.mean(L_AA) - np.mean(S_AA)
        score_BB = np.mean(L_BB) - np.mean(S_BB)
        score_AB = np.mean(L_AB) - np.mean(S_AB)
        
        return score_AA + score_BB + score_AB

    scores = df.groupby('subject_id').apply(subject_score)
    return float(scores.mean())
outcome: self_sim=0.0012 (var=0.0689) adversary_sim=-0.0471 (var=0.0417) welch_t=+0.726 p=0.4717 (N=25, alpha=0.01) -> reject

[4] rationale: We calculate two differences in the empirical probability of choosing Option A. 

Difference 1 (d1) contrasts Trial 4 and Trial 5. Under TTB, T4 predicts B and T5 predicts A, so d1 is negative. Under WADD, the difference in weighted sums is +1.7 for T4 and -1.0 for T5, so d1 is proportional to 2.7 * alpha.

Difference 2 (d2) contrasts Trial 1 and Trial 8. Under TTB, T1 predicts A and T8 predicts B, so d2 is positive. Under WADD, the difference in weighted sums is +3.5 for T1 and -0.4 for T8, so d2 is proportional to 3.9 * alpha.

By taking the product of d1 and d2 per subject, we obtain a metric that is strictly negative for TTB (negative * positive) and strictly positive for WADD (proportional to 10.53 * alpha^2, which is positive regardless of whether the subject's alpha is positive or negative). This completely factors out the variance of the unknown alpha parameter and cleanly discriminates the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    df = data.copy()
    
    # Create string keys for trial types (n_features=5)
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x])) + "_" + df['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    df['chose_A'] = (df['response'] == 0).astype(float)
    
    def subject_score(subj_data):
        p_A = subj_data.groupby('trial_type')['chose_A'].mean()
        
        # Difference 1: Trial 4 vs Trial 5
        # T4: A=[0,1,1,1,1] B=[1,0,0,0,0]. TTB predicts B. WADD score diff = +1.7 * alpha
        # T5: A=[0,1,0,0,0] B=[0,0,1,1,1]. TTB predicts A. WADD score diff = -1.0 * alpha
        p_T4 = p_A.get("01111_10000", 0.5)
        p_T5 = p_A.get("01000_00111", 0.5)
        d1 = p_T4 - p_T5
        
        # Difference 2: Trial 1 vs Trial 8
        # T1: A=[1,1,1,1,1] B=[0,0,0,0,0]. TTB predicts A. WADD score diff = +3.5 * alpha
        # T8: A=[1,0,0,1,1] B=[1,1,1,0,0]. TTB predicts B. WADD score diff = -0.4 * alpha
        p_T1 = p_A.get("11111_00000", 0.5)
        p_T8 = p_A.get("10011_11100", 0.5)
        d2 = p_T1 - p_T8
        
        # For TTB, d1 is negative and d2 is positive -> product is strictly negative.
        # For WADD, d1 and d2 have the same sign (both proportional to alpha) -> product is strictly positive.
        return d1 * d2

    scores = df.groupby('subject_id').apply(subject_score)
    return float(scores.mean())

outcome: self_sim=0.0026 (var=0.0020) adversary_sim=-0.0015 (var=0.0012) welch_t=+0.368 p=0.7145 (N=25, alpha=0.01) -> reject

[5] rationale: Under the Take-The-Best (TTB) model, the decision maker's probability of choosing Option A is identical across all trials where TTB predicts Option A. Trials 1, 2, 3, 5, 6, and 7 are all trials where TTB predicts Option A. Thus, for any subject following TTB, the variance of the empirical choice probabilities across these 6 trials should be very small, driven entirely by binomial sampling noise (~0.02). Conversely, under Contrarian WADD, the difference in weighted sums between Option A and Option B varies drastically across these 6 trials (e.g., +3.5 for Trial 1 vs -1.0 for Trial 5). This forces the true choice probabilities to vary significantly across these trials, resulting in a much higher sample variance. By calculating the within-subject variance of the choice probabilities across these specific 6 trials, we cleanly separate the models regardless of the sign of WADD's alpha parameter.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    df = data.copy()
    
    # Create string keys for trial types (n_features=5)
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x])) + "_" + df['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    df['chose_A'] = (df['response'] == 0).astype(float)
    
    def subject_score(subj_data):
        p_A = subj_data.groupby('trial_type')['chose_A'].mean()
        
        # Under TTB, the model predicts Option A for all 6 of these trials.
        # Because the TTB model outputs the exact same probability for any trial where it predicts A,
        # the true probability of choosing A is constant across these 6 trials for TTB.
        t1 = p_A.get("11111_00000", 0.5)
        t2 = p_A.get("10000_01000", 0.5)
        t3 = p_A.get("10101_01010", 0.5)
        t5 = p_A.get("01000_00111", 0.5)
        t6 = p_A.get("11000_10111", 0.5)
        t7 = p_A.get("01100_00011", 0.5)
        
        # For TTB, the variance of these empirical probabilities is purely due to binomial sampling noise.
        # For WADD, the underlying weighted sum differences vary wildly across these 6 trials (from -1.0 to 3.5),
        # so the variance of the empirical probabilities will be significantly higher.
        return np.var([t1, t2, t3, t5, t6, t7], ddof=1)

    scores = df.groupby('subject_id').apply(subject_score)
    return float(scores.mean())
outcome: self_sim=0.0211 (var=0.0002) adversary_sim=0.0211 (var=0.0003) welch_t=+0.013 p=0.99 (N=25, alpha=0.01) -> reject

[6] rationale: This metric evaluates a carefully balanced linear contrast of the empirical choice probabilities. For the TTB model, the 8 trials perfectly partition into two sets: those where TTB predicts A (Trials 1, 2, 3, 5, 6, 7) and those where TTB predicts B (Trials 4, 8). By centering the weights within each set to sum to exactly 0, the contrast completely cancels out any TTB signal or response bias, giving a rigorous expected value of 0 for TTB. For Contrarian WADD, the weights correspond to the underlying differences in weighted sums. Since the alpha parameter in Contrarian WADD has a negative mean (-0.5), this linear contrast will have a strong, non-zero negative expected value, cleanly and linearly discriminating the models without amplifying binomial noise through absolute values or products.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    df = data.copy()
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x])) + "_" + df['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    df['chose_A'] = (df['response'] == 0).astype(float)
    
    def subject_score(subj_data):
        p = subj_data.groupby('trial_type')['chose_A'].mean()
        
        T1 = p.get("11111_00000", 0.5)
        T2 = p.get("10000_01000", 0.5)
        T3 = p.get("10101_01010", 0.5)
        T4 = p.get("01111_10000", 0.5)
        T5 = p.get("01000_00111", 0.5)
        T6 = p.get("11000_10111", 0.5)
        T7 = p.get("01100_00011", 0.5)
        T8 = p.get("10011_11100", 0.5)
        
        # Contrast weights designed such that their sum is exactly 0 for trials where TTB predicts A
        # (T1, T2, T3, T5, T6, T7) and exactly 0 for trials where TTB predicts B (T4, T8).
        # Under TTB, the expected value of this contrast is strictly 0.
        # Under Contrarian WADD, the weights are proportional to the mean-centered differences 
        # in weighted sums, making the contrast highly sensitive to the `alpha` parameter.
        C = (3.05 * T1 - 0.35 * T2 + 0.25 * T3 
             + 1.05 * T4 - 1.45 * T5 - 1.45 * T6 
             - 0.05 * T7 - 1.05 * T8)
        
        return C

    scores = df.groupby('subject_id').apply(subject_score)
    return float(scores.mean())
outcome: self_sim=-0.0416 (var=0.3932) adversary_sim=0.1094 (var=0.3056) welch_t=-0.903 p=0.371 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    df = data.copy()\n    \n    # Convert binary rating arrays to string keys to safely identify trial types\n    df['trial_type'] = df['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x])) + \"_\" + df['option_b_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    df['chose_A'] = (df['response'] == 0).astype(float)\n    \n    # These weights are mathematically constructed to sum to exactly 0 within \n    # the two Take-The-Best (TTB) equivalence classes (trials where TTB predicts A,\n    # and trials where TTB predicts B). \n    # For TTB, the expected value of this linear contrast is rigorously 0.\n    # For Contrarian WADD, the weights are proportional to the mean-centered differences \n    # in weighted sums, maximizing the extracted signal.\n    weights = {\n        \"11111_00000\": 3.05,\n        \"10000_01000\": -0.35,\n        \"10101_01010\": 0.25,\n        \"01111_10000\": 1.05,\n        \"01000_00111\": -1.45,\n        \"11000_10111\": -1.45,\n        \"01100_00011\": -0.05,\n        \"10011_11100\": -1.05\n    }\n    \n    def subject_score(subj_data):\n        stats = subj_data.groupby('trial_type')['chose_A'].agg(['mean', 'count']).to_dict('index')\n        \n        contrast = 0.0\n        bias_correction = 0.0\n        \n        for t_type, w in weights.items():\n            if t_type in stats:\n                p_i = stats[t_type]['mean']\n                n_i = stats[t_type]['count']\n                \n                contrast += w * p_i\n                \n                # Compute the unbiased estimate of the binomial noise variance for this term\n                if n_i > 1:\n                    bias_correction += (w ** 2) * p_i * (1.0 - p_i) / (n_i - 1)\n                    \n        # Squaring the contrast ensures the WADD signal is strictly positive regardless of \n        # whether the subject's `alpha` parameter is positive or negative.\n        # Subtracting the bias_correction rigorously removes the positive bias introduced \n        # by binomial noise, keeping the TTB expected value centered at 0.\n        squared_signal = contrast ** 2 - bias_correction\n        \n        # Taking the square root compresses the heavy tails of the alpha parameter distribution,\n        # drastically reducing between-subject variance and stabilizing the t-statistic.\n        return float(np.sqrt(max(0.0, squared_signal)))\n\n    scores = df.groupby('subject_id').apply(subject_score)\n    return float(scores.mean())",
  "rationale": "This metric evaluates an optimal linear contrast of empirical choice probabilities designed to perfectly isolate the WADD signal from TTB. Under the TTB model, the 8 trial types partition into two equivalence classes (predicts A vs predicts B). The contrast weights are mathematically constrained to sum to exactly 0 within each class, guaranteeing that the expected value of the contrast is rigorously 0 for any subject following TTB, regardless of their lapse rate.\n\nBecause the WADD model's `alpha` parameter can be positive or negative, a simple linear contrast would have a mean near zero and massive between-subject variance. To solve this, we square the contrast per subject, making the WADD signal strictly positive. However, squaring an empirical contrast introduces a positive bias due to binomial sampling noise. We rigorously remove this bias by subtracting an exact, unbiased estimate of the binomial variance for each subject. Finally, we take the square root of the bias-corrected signal to compress the heavy tails of the alpha parameter's distribution. This yields a metric with a tight, near-zero distribution for TTB and a robust, strictly positive distribution for WADD, cleanly discriminating the theories."
}
```

## Usage

```json
{
  "prompt_token_count": 9770,
  "candidates_token_count": 1061,
  "total_token_count": 38266
}
```
